import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

'''
LLaMa-2
https://github.com/meta-llama/llama/
https://github.com/hkproj/pytorch-llama
'''


@dataclass
class ModelArgs:
    '''
    n_kv_heads:
        This is the number of key_value heads that should be used to implement Grouped Query Attention.
        If `n_kv_heads=n_heads`, the model will use Multi Head Attention (MHA),
        if `n_kv_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
        by mean-pooling all the original heads within that group. For more details checkout [this
        paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
        `num_attention_heads`.
    '''
    dim: int = 4096  # Dimension of the hidden representations.
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
    use_cache: bool = True  # use use_cache or not
    dropout: float = 0.0


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
        self.args = args
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_q_heads = args.n_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads  # dim of every head

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # Linear transformation for output.
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: Tensor, start_pos: int, freqs_complex: Tensor, mask: Optional[Tensor]):
        # x: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # -> (batch_size, seq_len, n_head_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # -> (batch_size, seq_len, n_head_q, head_dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, self.args.device)
        # -> (batch_size, seq_len, n_head_kv, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, self.args.device)

        if self.args.use_cache:
            # 保存
            self.cache_k[:batch_size, start_pos: start_pos + seq_len] = xk
            self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv
            # -> (batch_size, seq_len_kv, n_head_kv, head_dim)
            # ！！！这里是关键，也就是计算当前token，是拿到了之前的token
            # https://github.com/meta-llama/llama/issues/151
            xk = self.cache_k[:batch_size, 0: start_pos + seq_len]
            xv = self.cache_v[:batch_size, 0: start_pos + seq_len]

        # (batch_size, seq_len_kv, n_head_q, head_dim)
        # 因为 self.n_rep = self.n_q_heads // self.n_kv_heads
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # -> (batch_size, n_head_q, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        # -> (batch_size, n_head_q, seq_len_kv, head_dim)
        xk = xk.transpose(1, 2)
        # -> (batch_size, n_head_q, seq_len_kv, head_dim)
        xv = xv.transpose(1, 2)

        # (batch_size, n_head_q, seq_len, head_dim) @ (batch_size, n_head_q, head_dim, seq_len_kv)
        # -> (batch_size, n_head_q, seq_len, seq_len_kv)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (batch_size, n_head_q, seq_len, seq_len_kv)
        # -> (batch_size, n_head_q, seq_len, seq_len_kv)
        scores = self.dropout(F.softmax(scores.float(), dim=-1).type_as(xq))

        # (batch_size, n_head_q, seq_len, seq_len_kv) @ (batch_size, n_head_q, seq_len_kv, head_dim)
        # -> (batch_size, n_head_q, seq_len, head_dim)
        output = torch.matmul(scores, xv)
        # (batch_size, n_head_q, seq_len, head_dim) -> (batch_size, seq_len, n_head_q, head_dim)
        # -> (batch_size, seq_len, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 这一堆计算能否简化
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
        # 为什么设计成这样? 看 GLU Variants Improve Transformer
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
        # Attention, FeedForward 都是输入：(batch_size, seq_len, dim)
        # 输出：(batch_size, seq_len, dim)
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(args)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: Tensor, start_pos: int, freqs_complex: Tensor, mask: Optional[Tensor]):
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
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(DecoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
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
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=self.args.device), diagonal=1)
            mask = torch.hstack([torch.zeros((seq_len, start_pos), device=self.args.device), mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
