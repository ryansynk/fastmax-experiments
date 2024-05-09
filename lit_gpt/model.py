# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import einsum
from typing_extensions import Self

from lit_gpt.config import Config

from torch.cuda.amp import autocast
from functools import partial
from fast_transformers.causal_product import CausalDotProduct
from contextlib import contextmanager

from attention_mechanisms.fastmax import fastmax
from attention_mechanisms.fastmax_hack import fastmax_hack

import fastmax_cuda


class FASTMultiHeadAttention_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        drop_noise,
        rpe_matrix=None,
        mask=False,
        dropout=0.0,
        normalize=False,
        temperature=1.0,
    ):
        b = 0
        if len(q.shape) == 4:
            b = q.shape[0]
            q = q.reshape(
                (q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
            )  # (b,h,n,d) -> (b*h,n,d)
            k = k.reshape(
                (k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
            )  # (b,h,n,d) -> (b*h,n,d)
            v = v.reshape(
                (v.shape[0] * v.shape[1], v.shape[2], v.shape[3])
            )  # (b,h,n,d) -> (b*h,n,d)
            drop_noise = drop_noise.reshape(
                (
                    drop_noise.shape[0] * drop_noise.shape[1],
                    drop_noise.shape[2],
                    drop_noise.shape[3],
                )
            )  # (b,h,n,d) -> (b*h,n,d)
        elif len(q.shape) != 3:
            print(
                "q, k, and v should be either 3 or 4 dimensional tensors. If 3D: (b*h,n,d), if 4D: (b,h,n,d)."
            )

        if rpe_matrix is None:
            print(
                "Relative Positional Encoding must be given. Send a 2*n-1 by d matrix of all zeros if you don't want to use RPE."
            )

        q = q.permute(1, 2, 0).contiguous()  # (b*h,n,d) -> (n,d,b*h)
        k = k.permute(1, 2, 0).contiguous()  # (b*h,n,d) -> (n,d,b*h)
        v = v.permute(1, 2, 0).contiguous()  # (b*h,n,d) -> (n,d,b*h)
        drop_noise = drop_noise.permute(1, 2, 0).contiguous()  # (b*h,n,d) -> (n,d,b*h)

        breakpoint()
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        o = fastmax_cuda.forwardpass(
            q, k, v, drop_noise, rpe_matrix, mask, dropout, normalize, temperature
        )
        ctx.save_for_backward(q, k, v, o)
        ctx.mask = mask
        ctx.b = b
        ctx.t = temperature
        o = (
            o[:, : q.shape[1], :].permute(2, 0, 1).contiguous()
        )  # (n,d,b*h) -> (b*h,n,d)
        if b != 0:
            o = o.reshape(
                (b, int(o.shape[0] / b), o.shape[1], o.shape[2])
            )  # (b*h,n,d) -> (b,h,n,d)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o = ctx.saved_tensors
        mask = ctx.mask
        b = ctx.b
        t = ctx.t

        if b != 0:
            grad_output = grad_output.reshape(
                (
                    grad_output.shape[0] * grad_output.shape[1],
                    grad_output.shape[2],
                    grad_output.shape[3],
                )
            ).contiguous()
        grad_output = grad_output.permute(
            1, 2, 0
        ).contiguous()  # (b*h,n,d) -> (n,d,b*h)
        gradq, gradk, gradv = fastmax_cuda.backwardpass(q, k, v, o, grad_output, mask)

        gradq = gradq.permute(2, 0, 1).contiguous()  # (n,d,b*h) -> (b*h,n,d)
        gradk = gradk.permute(2, 0, 1).contiguous()  # (n,d,b*h) -> (b*h,n,d)
        gradv = gradv.permute(2, 0, 1).contiguous()  # (n,d,b*h) -> (b*h,n,d)

        if b != 0:
            gradq = gradq.reshape(
                (b, int(gradq.shape[0] / b), gradq.shape[1], gradq.shape[2])
            ).contiguous()
            gradk = gradk.reshape(
                (b, int(gradk.shape[0] / b), gradk.shape[1], gradk.shape[2])
            ).contiguous()
            gradv = gradv.reshape(
                (b, int(gradv.shape[0] / b), gradv.shape[1], gradv.shape[2])
            ).contiguous()
        return gradq, gradk / t, gradv, None, None, None, None, None, None


class FASTMultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(FASTMultiHeadAttention, self).__init__()

    def forward(
        self,
        q,
        k,
        v,
        drop_noise,
        rpe_matrix=None,
        mask=False,
        dropout=0.0,
        normalize=False,
        temperatue=1.0,
    ):
        return FASTMultiHeadAttention_Function.apply(
            q, k, v, drop_noise, rpe_matrix, mask, dropout, normalize, temperatue
        )


def rpe_matrix_creator(n, d, device, dtype, structured=True, is_zero=False):
    """
    Creates the relative positional encoding matrix
    Inputs: (assuming query is a (b,h,n,d) or (b*h,n,d) tensor)
      - n (int): number of tokens
      - d (int): dimesion/channel per head
      - data type: must be torch.float32. This input is used to make sure the datatype used by the attention head is torch.float32.
      - Structured (bool): if True, produces sin/cos based RPE, and randomized matrx otherwise.
    Output:
      - rpe: a (2*n-1,d) matrix.
    """
    if dtype != torch.float32:
        print("The data type must be float32 in order for Fastmax to work")
    if structured:
        pe_positive = torch.zeros(n, d, device=device, dtype=dtype)
        pe_negative = torch.zeros(n, d, device=device, dtype=dtype)
        position = torch.arange(0, n, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2, device=device, dtype=dtype) * -(math.log(10000.0) / d)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0])
        pe_negative = pe_negative[1:]
        rpe = torch.cat([pe_positive, pe_negative], dim=0)
    else:
        if is_zero:
            rpe = torch.zeros(0, 1, size=(2 * n - 1, d), device=device, dtype=dtype)
        else:
            rpe = torch.normal(0, 1, size=(2 * n - 1, d), device=device, dtype=dtype)
    return rpe


@contextmanager
def null_context():
    yield


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}"
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache()

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * (self.config.n_embd**0.5)

        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = (
            None
            if config.shared_attention_norm
            else config.norm_class(config.n_embd, eps=config.norm_eps)
        )
        self.mlp = config.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_1 = self.norm_1(x)
        h = self.attn(n_1, cos, sin, mask, input_pos)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(n_2) + h + x
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            x = h + x
            x = self.mlp(self.norm_2(x)) + x
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(
            B, T, self.config.n_query_groups, total_qkv, self.config.head_size
        )
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (
            input_pos is None or self.config.n_query_groups != 1
        ):
            k = k.expand(
                B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
            )
            v = v.expand(
                B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
            )

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        attn_alg = self.config.attn_alg
        if isinstance(attn_alg, str):
            pass
        elif isinstance(attn_alg, tuple):
            attn_alg = attn_alg[0]
        else:
            raise ValueError(f"Attention algorithm {attn_alg} has a type problem")

        if attn_alg == "quadratic":
            y = self.scaled_dot_product_attention(q, k, v, mask)
        elif attn_alg == "performer":
            y = self.performer_attention(q, k, v, input_pos)
        elif attn_alg == "linearmax":
            y = self.linearmax(q, k, v, input_pos)
        elif attn_alg == "fastmax":
            y = self.fastmax(q, k, v, input_pos)
        elif attn_alg == "fastmax_cuda":
            y = self.fastmax_cuda(q, k, v)
        else:
            raise ValueError(f"Attention algorithm {attn_alg} not supported")

        y = y.reshape(
            B, T, self.config.head_size * self.config.n_head
        )  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def linearmax(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        mask = True
        if input_pos is not None:
            # We are using KVCache, so we are at inference time and don't need the mask
            mask = False
        # q = q.cpu()
        # k = k.cpu()
        # v = v.cpu()
        # o = fastmax_hack(q, k, v, p=1, mask=mask)
        # o = o.cuda()
        o = fastmax_hack(q, k, v, p=1, mask=mask)
        return o

    def fastmax(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        mask = True
        if input_pos is not None:
            # We are using KVCache, so we are at inference time and don't need the mask
            mask = False
        q = q.cpu()
        k = k.cpu()
        v = v.cpu()
        o = fastmax(q, k, v, p=2, mask=mask)
        o = o.cuda()
        return o

    def fastmax_cuda(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # the inputs of fastmax are query, key, and value (q,k,v) in shape of  4-dimensional tensors (b, h, n, d); i.e. (batch, head, token length, dimension/channel per head)
        fastmax = FASTMultiHeadAttention()

        mask = True
        dropout = 0.0  # between 0 and 1
        normalize = True
        temperatue = 1.0
        a0 = 1.0
        a1 = 1.0
        a2 = 0.5
        lim = 1.0

        # NOTE: If you're performing cross attention, using relative positional encoding (RPE) wont make sense. To have the RPE mastrix be zero, set the flags below as structured = False, is_zero = True
        # rpe_matrix = rpe_matrix_creator(k.shape[-2], q.shape[-1], q.device, q.dtype, structured=False, is_zero=True)
        rpe_matrix = rpe_matrix_creator(
            k.shape[-2],
            q.shape[-1],
            q.device,
            torch.float32,
            structured=True,
            is_zero=False,
        )
        drop_noise = torch.normal(
            0, 1, size=(q.shape), dtype=torch.float32, device=q.device
        )
        o = fastmax(
            q,
            k,
            v,
            drop_noise,
            rpe_matrix,
            mask,
            dropout,
            normalize,
            temperatue,
            a0,
            a1,
            a2,
            lim,
        )
        return o

    def performer_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        eps=1e-6,
    ) -> torch.Tensor:
        autocast_enabled = torch.is_autocast_enabled()
        # is_half = isinstance(q, torch.cuda.HalfTensor)
        # if is_half: assert APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
        cuda_context = (
            null_context if not autocast_enabled else partial(autocast, enabled=False)
        )

        # print(autocast_enabled)
        k = k[:, :, : q.size(dim=2), :]
        v = v[:, :, : q.size(dim=2), :]

        k_cumsum = k.cumsum(dim=-2) + eps
        D_inv = 1.0 / torch.einsum("...nd,...nd->...n", q, k_cumsum.type_as(q))
        Q = q.float()
        K = k.float()
        V = v.float()
        causal_dot_product_fn = CausalDotProduct.apply
        # q, k, v = map(lambda t: t.float(), (q, k, v))
        # with torch.autocast('cuda', torch.bfloat16, enabled=True):
        #     print(q.dtype, k.dtype, v.dtype)
        #     print(torch.mm(torch.randn(3,4), torch.randn(4,5)).dtype)
        #     out = causal_dot_product_fn(q + eps, k + eps, v + eps)
        out = causal_dot_product_fn(Q, K, V)
        out = out.to(torch.float16)
        out = torch.einsum("...nd,...n->...nd", out, D_inv)
        return out

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.gelu(x_fc_1) * x_fc_2
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(
            router, self.config.n_expert_per_token
        )  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(
            self.config.n_expert, device=x.device
        )
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False
        )

    def forward(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(
    max_seq_length: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)
