import math
import time
import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sentencepiece import SentencePieceProcessor

'''
https://github.com/meta-llama/llama/
https://github.com/hkproj/pytorch-llama
'''


@dataclass
class ModelArgs:
    dim: int = 4096  # d_model
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # Later set in the build method
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, base: float = 10000.0):
    assert head_dim % 2 == 0, "dimension must be divisible by 2"
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (head_dim / 2)
    theta = 1.0 / (base ** (theta_numerator / head_dim)).to(device)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Shape: (seq_len) outer_product* (head_dim / 2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # Shape: (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: Tensor, freqs_complex: Tensor, device: str):
    # Shape: (batch_size, seq_len, n_head, head_dim) -> (batch_size, seq_len, n_head, head_dim/2)
    # n_head * head_dim = dim
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Shape: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Shape: (batch_size, seq_len, n_head, head_dim/2) * (1, seq_len, 1, head_dim/2) 
    # = (batch_size, seq_len, n_head, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # Shape: (batch_size, seq_len, n_head, head_dim/2) -> (batch_size, seq_len, n_head, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # Shape: (batch_size, seq_len, n_head, head_dim/2, 2) -> (batch_size, seq_len, n_head, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # (batch_size, seq_len, n_kv_head, 1, head_dim) ->
    # (batch_size, seq_len, n_kv_head, n_rep, head_dim) ->
    # (batch_size, seq_len, n_kv_head * n_rep, head_dim)
    return (x[:, :, :, None, :].
            expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim).
            reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
            )


def sample_top_p(probs, p):
    # (batch_size, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (batch_size, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (batch_size, vocab_size)
    mask = probs_sum - probs_sort > p
    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor):
        # (batch_size, seq_len, dim) * (batch_size, seq_len, 1) = (batch_size, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        # (dim) * (batch_size, seq_len, dim) = (batch_size, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.device = args.device

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # Linear transformation for output.
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: Tensor, start_pos: int, freqs_complex: Tensor, mask: Optional[Tensor]):
        # x: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # -> (batch_size, seq_len, n_head_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # -> (batch_size, seq_len, n_head_q, head_dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, self.device)
        # -> (batch_size, seq_len, n_head_kv, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, self.device)

        # 保存
        self.cache_k[:batch_size, start_pos: start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv

        # -> (batch_size, seq_len_kv, n_head_kv, head_dim)
        # 这里是关键，也就是计算当前token，是拿到了之前的token
        keys = self.cache_k[:batch_size, 0: start_pos + seq_len]
        values = self.cache_v[:batch_size, 0: start_pos + seq_len]

        # (batch_size, seq_len_kv, n_head_q, head_dim)
        # 因为 self.n_rep = self.n_heads_q // self.n_kv_heads
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # -> (batch_size, n_head_q, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        # -> (batch_size, n_head_q, seq_len_kv, head_dim)
        keys = keys.transpose(1, 2)
        # -> (batch_size, n_head_q, seq_len_kv, head_dim)
        values = values.transpose(1, 2)

        # (batch_size, n_head_q, seq_len, head_dim) @ (batch_size, n_head_q, head_dim, seq_len_kv)
        # -> (batch_size, n_head_q, seq_len, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask  # (batch_size, n_head_q, seq_len, seq_len_kv)
        # -> (batch_size, n_head_q, seq_len, seq_len_kv)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (batch_size, n_head_q, seq_len, seq_len_kv) @ (batch_size, n_head_q, seq_len_kv, head_dim)
        # -> (batch_size, n_head_q, seq_len, head_dim)
        output = torch.matmul(scores, values)
        # (batch_size, n_head_q, seq_len, head_dim) -> (batch_size, seq_len, n_head_q, head_dim)
        # -> (batch_size, seq_len, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 怎么计算这么复杂
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: Tensor):
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, hidden_dim)
        swish = F.silu(self.w1(x))
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, hidden_dim)
        x_V = self.w3(x)
        # (batch_size, seq_len, hidden_dim) * (batch_size, seq_len, hidden_dim)
        # -> (batch_size, seq_len, hidden_dim)
        x = swish * x_V
        # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, dim)
        x = self.w2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        # Attention, FeedForward 都是输入：(batch_size, seq_len, dim)
        # 输出：(batch_size, seq_len, dim)
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: Tensor, start_pos: int, freqs_complex: Tensor, mask: Optional[Tensor], ):
        # (batch_size, seq_len, dim) + (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex, mask)
        # (batch_size, seq_len, dim) + (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2,
                                                              self.args.device)

    def forward(self, tokens: Tensor, start_pos: int):
        # (batch_size, seq_len)
        batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        mask = None
        # 当第一次读入prompt时，prompt较长，可以加mask一次计算，以后读入单个token，就不需要mask
        if seq_len > 1:
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"),
                                         device=self.args.device),
                              diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seq_len, start_pos), device=self.args.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


class LlamaForCompletion(nn.Module):
    def __init__(self, model: LlamaModel, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, max_batch_size: int, max_seq_len: int = 2048,
              device: str = 'cpu'):

        checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
        ckpt_path = checkpoints[0]
        print(f'Loading checkpoint "{ckpt_path}"')
        prev_time = time.time()
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        model = LlamaModel(model_args).to(device)

        # !!!!!! The only unmatched key in the checkpoint is rope.freqs. Remove it
        del checkpoint['rope.freqs']
        prev_time = time.time()
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return LlamaForCompletion(model, tokenizer, model_args)

    def forward(self, prompts: list[str],
                temperature: float = 0.6,
                top_p: float = 0.9,
                max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
                         for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, \
            f"batch size must be less than or equal to {self.args.max_batch_size}"
        min_prompt_len = min(len(prompt) for prompt in prompt_tokens)
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, \
            f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)

        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        prompt_tokens_mask = tokens != pad_id
        cur_iterator = tqdm(range(min_prompt_len, total_len), desc="Generating tokens")
        prev_pos = 0
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens, out_text = [], []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return out_tokens, out_text


class LlamaForSequenceClassification:
    pass
