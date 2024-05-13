import torch
import torch.nn.functional as F
from torch import nn, Tensor
import json
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from xformers.ops.fmha.attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)
from xformers.ops.fmha import memory_efficient_attention
from llama import FeedForward, RMSNorm, precompute_freqs_cis, apply_rotary_embeddings

'''
https://www.youtube.com/watch?v=UiX8K-xBUpE
https://github.com/hkproj/mistral-src-commented
https://github.com/hkproj/mistral-llm-notes/
mixtral的kv_cache没看懂
'''


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8

    head_dim: int = 128  # dim / n_heads
    vocab_size: int = 32000
    hidden_dim: int = 14336
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 8192

    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2

    sliding_window: Optional[int] = 4096
    device: str = None


@dataclass
class RotatingCacheInputMetadata:
    # rope absolute positions
    positions: torch.Tensor
    # which elements in the sequences need to be cached
    to_cache_mask: torch.Tensor
    # how many elements are cached per sequence
    cached_elements: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias  # Mask for the attention
    seqlens: List[int]


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )


def interleave_list(l1: List[Tensor], l2: List[Tensor]):
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


def unrotate(cache: Tensor, seqlen: int) -> Tensor:
    # seqlen is the total number of tokens cached so far, including the overwritten one.
    # This is needed to calculate the rotation point of the cache
    # (Sliding_Window_Size, Num_Heads, Head_Dim)
    assert cache.ndim == 3
    position = seqlen % cache.shape[0]  # This is the pivot point around which we need to rotate the cache
    # If the total sequence length so far is smaller than the cache size,
    # then just return the first seqlen elements, as the cache didn't have any rotations yet
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        # Select the unrotated elements from the cache around the pivot point
        return torch.cat([cache[position:], cache[:position]], dim=0)


def repeat_kv(keys: Tensor, values: Tensor, repeats: int, dim: int):
    # (Seq, N_Heads_KV, Head_Dim) --> (Seq, N_Heads, Head_Dim)
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class CacheView:
    def __init__(self, cache_k: Tensor, cache_v: Tensor, metadata: RotatingCacheInputMetadata, kv_seqlens: Tensor):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: Tensor, xv: Tensor):
        """
        to_cache_mask masks the last [sliding_window] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        # (Max_Batch_Size, Sliding_Window_Size, N_Heads_KV, Head_Dim)
        # -> (Max_Batch_Size * Sliding_Window_Size, N_Heads_KV, Head_Dim)
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        # (Max_Batch_Size, Sliding_Window_Size, N_Heads_KV, Head_Dim)
        # -> (Max_Batch_Size * Sliding_Window_Size, N_Heads_KV, Head_Dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)
        # Copies from the xk and xv tensors to the cache tensors,
        # based on the cache positions and the items to cache (to_cache_mask)
        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3  # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(Seq, N_Heads_KV, Head_Dim)]
        # (Seq1+Seq2+Seq3, N_Heads_KV, Head_Dim)
        # -> [(Seq1, N_Heads_KV, Head_Dim), (Seq2, N_Heads_KV, Head_Dim), (Seq3, N_Heads_KV, Head_Dim)]
        xk = torch.split(xk, self.metadata.seqlens)
        # (Seq1+Seq2+Seq3, N_Heads_KV, Head_Dim)
        # -> [(Seq1, N_Heads_KV, Head_Dim), (Seq2, N_Heads_KV, Head_Dim), (Seq3, N_Heads_KV, Head_Dim)]
        xv = torch.split(xv, self.metadata.seqlens)
        assert len(xk) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Order elements in cache by position by unrotating
        # Currently cached elements, already unrotated, one for each prompt
        cache_k = [unrotate(t, s) for t, s in zip(self.cache_k, self.kv_seqlens)]
        # Currently cached elements, already unrotated, one for each prompt
        cache_v = [unrotate(t, s) for t, s in zip(self.cache_v, self.kv_seqlens)]
        # Appends the incoming keys and values to the currently cached elements (one for each prompt)
        interleaved_k = interleave_list(cache_k, xk)
        # Appends the incoming keys and values to the currently cached elements (one for each prompt)
        interleaved_v = interleave_list(cache_v, xv)

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def sliding_window(self):
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[:len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[:len(self.kv_seqlens)]

    @property
    def prefill(self):
        return self.metadata.prefill

    @property
    def mask(self):
        return self.metadata.mask


class RotatingBufferCache:
    """
    This is an example that implements a less naive rotating buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """

    def __init__(self, n_layers: int, max_batch_size: int, sliding_window: int, n_kv_heads: int, head_dim: int):
        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim  # model_dim / n_heads

        self.cache_k = torch.empty((n_layers, max_batch_size, sliding_window, n_kv_heads, head_dim))
        self.cache_v = torch.empty((n_layers, max_batch_size, sliding_window, n_kv_heads, head_dim))
        # holds the valid length for each batch element in the cache
        self.kv_seqlens = None

    def get_view(self, layer_id: int, metadata: RotatingCacheInputMetadata) -> CacheView:
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens)

    def reset(self):
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int):
        self.kv_seqlens = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    @property
    def device(self):
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype):
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]):
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> RotatingCacheInputMetadata:
        """
            inpput = seqlens [5,7,2] // seqpos [0, 1, 3] // sliding_window 3
            --> only cache last 3 tokens in each sequence
            - to_cache_mask = [0 0 1 1 1 | 0 0 0 0 1 1 1 | 1 1]
            - cached_elements = [3 | 3 | 2]
            --> absolute positions are used for rope
            - positions = [0 1 2 3 4 | 1 2 3 4 5 6 7 | 3 4]
            --> cache positions are positions cache_masked, modulo sliding_window + batch_idx * sliding_window
            - cache_positions = [2 0 1 | 5 3 4 | 6 7]
        """
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))
        assert len(seqlens) == len(
            self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        seqpos = self.kv_seqlens.tolist()  # Indicates the total length seen by the cache so far (including the overwritten elements) for each prompt

        assert len(seqlens) > 0, seqlens

        # [True] if the token position belongs to the last `sliding_window` positions of the sequence. It is always True unless the chunk size is bigger than the sliding window
        # Indicates which items in the sequence should be cached (the last `sliding_window` tokens of each sequence)
        masks = [
            [x >= seqlen - self.sliding_window for x in range(seqlen)]
            for seqlen in seqlens
            # The sequence length of each input in the batch (so we can understand which token belongs to which prompt)
        ]

        # Indicates which items in the sequence should be cached (the last `sliding_window` tokens of each sequence)
        # Concatenate all the masks of each prompt in the batch
        to_cache_mask = torch.tensor(sum(masks, []), device=self.device, dtype=torch.bool)

        # Number of elements in the mask == True
        cached_elements = torch.tensor([sum(mask) for mask in masks], device=self.device, dtype=torch.long)

        # The position of each token in the prompt (all concatenated). It may not start from zero (because for example the first chunk may be 5 tokens and we are now processing the second chunk)
        positions = torch.cat([torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]).to(
            device=self.device, dtype=torch.long)

        # The index of the batch to which each token (in the concatenated list) belongs to.
        batch_idx = torch.tensor(sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []), device=self.device,
                                 dtype=torch.long)

        # Where each token should be placed in the cache (based on the position in the prompt and the batch index)
        cache_positions = positions % self.sliding_window + batch_idx * self.sliding_window

        # Indicates if it is the first prefill (only True on the first chunk)
        first_prefill = seqpos[0] == 0
        # Indicates if it is a subsequent prefill (True from second chunk onwards), but False when generating tokens.
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)

        if first_prefill:
            # For first chunk of prompt. It creates an attention mask that is causal for each prompt and also local based on the sliding window size
            # https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.fmha.attn_bias.BlockDiagonalMask + local attention based on the sliding window
            assert all([pos == 0 for pos in seqpos]), (seqpos)
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(self.sliding_window)
        elif subsequent_prefill:
            # For subsequent chunks of prompt
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,  # Size of the query
                kv_seqlen=[s + cached_s.clamp(max=self.sliding_window).item() for (s, cached_s) in
                           zip(seqlens, self.kv_seqlens)]
                # The total number of keys and values will be the incoming sequence length + the cached elements
            ).make_local_attention_from_bottomright(self.sliding_window)
        else:  # For token generation
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,  # Size of the query
                kv_padding=self.sliding_window,
                kv_seqlen=(self.kv_seqlens + cached_elements).clamp(max=self.sliding_window).tolist()
                # The total number of keys and values will be the incoming sequence length + the cached elements
            )

        return RotatingCacheInputMetadata(positions=positions,
                                          to_cache_mask=to_cache_mask,
                                          cached_elements=cached_elements,
                                          cache_positions=cache_positions[to_cache_mask],
                                          prefill=first_prefill or subsequent_prefill,
                                          mask=mask,
                                          seqlens=seqlens,
                                          )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads
        # self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(self, x: Tensor, freqs_cis: Tensor, cache: Optional[CacheView]) -> Tensor:
        seqlen_sum, _ = x.shape  # (Seq, Dim)

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)  # (Seq, Dim) --> (Seq, Dim)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)  # (Seq, Dim) --> (Seq, N_Heads, Head_Dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)  # (Seq, Dim) --> (Seq, N_Heads_KV, Head_Dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)  # (Seq, Dim) --> (Seq, N_Heads_KV, Head_Dim)
        # -> (batch_size, seq_len, n_head_q, head_dim)
        xq = apply_rotary_embeddings(xq, freqs_cis, self.args.device)
        # -> (batch_size, seq_len, n_head_kv, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_cis, self.args.device)
        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # (Seq, N_Heads, Head_Dim), (Seq, N_Heads_KV, Head_Dim)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            # Appends the incoming keys and values to the currently cached keys and values
            # (because we need to use them for attention)
            key, val = cache.interleave_kv(xk, xv)
            # Add the incoming keys and values to the cache in the positions indicated by metadata.cache_positions
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value  # Retrieve the cached keys and values (including the newly added ones)
            key = key.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim)
            val = val.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim)

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        # Output: (B=1, Seq, N_Heads, Head_Dim)
        output = memory_efficient_attention(xq, key, val, None if cache is None else cache.mask)
        # (B=1, Seq, N_Heads, Head_Dim) --> (Seq, N_Heads * Head_Dim) --> (Seq, Dim)
        return self.wo(output.view(seqlen_sum, self.n_heads * self.head_dim))


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_experts_per_tok):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs: Tensor):
        gate_logits = self.gate(inputs)

        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        # Apply the softmax to the logits AFTER selecting the top-k,
        # this makes comparison with different hyperparams consitent.
        # Because even if we change the total number of experts or the number of experts per token,
        # the sum of the weights will still be 1 for each token.
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for current_expert_index, current_expert in enumerate(self.experts):
            # For each expert, select which token it will be applied to.
            token_index, token_expert_index = torch.where(selected_experts == current_expert_index)
            # Apply the expert to the selected tokens weighting it by the logits (post-softmax) computed above .
            results[token_index] += (weights[token_index, token_expert_index, None] *
                                     current_expert(inputs[token_index]))
        return results


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.feed_forward: nn.Module
        if args.use_moe:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(args.num_experts)],
                gate=nn.Linear(args.dim, args.num_experts, bias=False),
                num_experts_per_tok=args.num_experts_per_tok
            )
        else:
            self.feed_forward = FeedForward(args=args)

    def forward(self, x: Tensor, freqs_cis: Tensor, cache: Optional[CacheView]) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, pipeline_rank: int = 0, num_pipeline_ranks: int = 1):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        theta = 1000000.0 if self.args.sliding_window is None else 10000.0
        self.freqs_cis = precompute_freqs_cis(
            self.args.head_dim, 128_000, args.device, theta
        )

        # pipeline 相关
        assert self.vocab_size > 0
        assert pipeline_rank < num_pipeline_ranks, (pipeline_rank, num_pipeline_ranks)
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        # Modules specific to some ranks:
        self.tok_embeddings: Optional[nn.Embedding] = None
        self.norm: Optional[RMSNorm] = None
        self.output: Optional[nn.Linear] = None
        if pipeline_rank == 0:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        if pipeline_rank == num_pipeline_ranks - 1:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        # Initialize all layers but slice off those not of this rank.
        layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        num_layers_per_rank = math.ceil(args.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(args.n_layers, offset + num_layers_per_rank)
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})
        self.n_local_layers = len(self.layers)

    def forward_partial(self, input_ids: Tensor, seq_lens: List[int],
                        cache: Optional[RotatingBufferCache] = None) -> Tensor:
        """Local forward pass.
        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        assert (len(seq_lens) <= self.args.max_batch_size), \
            f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seq_lens)}"
        (num_toks,) = input_ids.shape
        assert sum(seq_lens) == num_toks, (sum(seq_lens), num_toks)
        if cache is not None:
            # Generate the attention mask based on the current stage:
            # first pre-fill, subsequent pre-fill or token generation.
            input_metadata = cache.get_input_metadata(seq_lens)
        else:
            # If we do not use the cache, then just return the positions of the tokens to be used for RoPE
            input_metadata = SimpleInputMetadata.from_seqlens(seq_lens, self.device)

        if self.pipeline_rank == 0:
            # Only the first GPU will take care of the embeddings
            assert self.tok_embeddings is not None
            h = self.tok_embeddings(input_ids)  # Transform the tokens into embeddings
        else:
            h = torch.empty(num_toks, self.args.dim, device=self.args.device)
            # Subsequent GPUs will receive the embeddings from the previous GPU
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        freqs_cis = self.freqs_cis[input_metadata.positions]

        # Apply each layer iteratively
        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                # Retrieves the KV cache for the current layer
                cache_view = cache.get_view(local_layer_id, input_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        if cache is not None:
            cache.update_seqlens(
                seq_lens)  # Updates the total sequence length so far seen by the cache among all the iterations
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # After all the layers for the current GPU have been applied, send the output to the next GPU
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            return h
        else:
            # Last rank has a final normalization step.
            assert self.norm is not None
            return self.norm(h)

    def forward(self, input_ids: Tensor, seqlens: List[int], cache: Optional[RotatingBufferCache] = None) -> Tensor:
        # The sequence length of each input in the batch (so we can understand which token belongs to which prompt)
        h = self.forward_partial(input_ids, seqlens, cache=cache)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # ignore the intermediate activations as we'll get the final output from
            # the last stage
            outs = torch.empty(h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype)
        else:
            assert self.output is not None
            outs = self.output(h)  # Apply the output linear projection of the embeddings to the vocabulary size
        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)
        return outs.float()

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_to_load = {}
        skipped = set([])
        for k, v in state_dict.items():
            if k.startswith("tok_embeddings"):
                if self.pipeline_rank == 0:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("norm") or k.startswith("output"):
                if self.pipeline_rank == self.num_pipeline_ranks - 1:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("layers"):
                layer_id = k.split(".")[1]
                if layer_id in self.layers:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            else:
                raise ValueError(f"Unexpected key {k}")
        assert set(state_dict.keys()) == skipped.union(set(state_to_load.keys()))
        super().load_state_dict(state_to_load, *args, **kwargs)

    @staticmethod
    def from_folder(folder: Path, max_batch_size: int = 1, num_pipeline_ranks: int = 1, device="cuda",
                    dtype=torch.float16) -> "Transformer":
        with open(folder / "params.json", "r") as f:
            model_args = ModelArgs.from_dict(json.load(f))
        model_args.max_batch_size = max_batch_size
        if num_pipeline_ranks > 1:
            pipeline_rank = torch.distributed.get_rank()
        else:
            pipeline_rank = 0
        with torch.device("meta"):
            model = Transformer(
                model_args,
                pipeline_rank=pipeline_rank,
                num_pipeline_ranks=num_pipeline_ranks,
            )
        loaded = torch.load(str(folder / "consolidated.00.pth"), mmap=True)
        model.load_state_dict(loaded, assign=True)
        return model.to(device=device, dtype=dtype)
