from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .rope import RotaryEmbedding


def sdpa_with_flattened_batch(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0
) -> Tensor:


    q_shape = q.shape
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])
    out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)

    return out.view(q_shape)


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor | int] = None,
    rope: Optional[RotaryEmbedding] = None,
) -> Tensor:


    if isinstance(attn_mask, int):
        assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
        assert rope is None, "Rotary position embedding is not supported with attn_mask as int"


    *batch_shape, tgt_len, embed_dim = query.shape
    src_len = key.shape[-2]

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"


    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)


    q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
    k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
    v = v.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)


    if rope is not None:
        q = rope.rotate_queries_or_keys(q)
        k = rope.rotate_queries_or_keys(k)


    if not training:
        dropout_p = 0.0

    if isinstance(attn_mask, int):
        cut_pos = attn_mask


        attn_output = torch.empty(*batch_shape, tgt_len, embed_dim, device=query.device, dtype=query.dtype)


        q_left = q[..., :cut_pos, :]
        k_left = k[..., :cut_pos, :]
        v_left = v[..., :cut_pos, :]

        attn_left = sdpa_with_flattened_batch(q_left, k_left, v_left, dropout_p=dropout_p)
        attn_left = attn_left.transpose(-3, -2).contiguous().view(*batch_shape, cut_pos, embed_dim)
        attn_output[..., :cut_pos, :] = F.linear(attn_left, out_proj_weight, out_proj_bias)


        if cut_pos < tgt_len:
            q_right = q[..., cut_pos:, :]
            attn_right = sdpa_with_flattened_batch(q_right, k_left, v_left, dropout_p=dropout_p)
            attn_right = attn_right.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len - cut_pos, embed_dim)
            attn_output[..., cut_pos:, :] = F.linear(attn_right, out_proj_weight, out_proj_bias)
    else:

        correct_2d_shape = (tgt_len, src_len)
        correct_nd_shape = (*batch_shape, num_heads, tgt_len, src_len)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                if attn_mask.shape != correct_2d_shape:
                    raise ValueError(f"2D attn_mask should have shape {correct_2d_shape}, but got {attn_mask.shape}")
                attn_mask = attn_mask.expand(*batch_shape, num_heads, tgt_len, src_len)
            elif attn_mask.dim() == len(correct_nd_shape):
                if attn_mask.shape != correct_nd_shape:
                    raise ValueError(
                        f"{len(correct_nd_shape)}D attn_mask should have shape {correct_nd_shape}, "
                        f"but got {attn_mask.shape}"
                    )
            else:
                raise ValueError(f"attn_mask must be 2D or {len(correct_nd_shape)}D, got {attn_mask.dim()}D")


        if key_padding_mask is not None:
            if key_padding_mask.shape != (*batch_shape, src_len):
                raise ValueError(
                    f"key_padding_mask should have shape {(*batch_shape, src_len)}, but got {key_padding_mask.shape}"
                )
            key_padding_mask = key_padding_mask.view(*batch_shape, 1, 1, src_len).expand(
                *batch_shape, num_heads, tgt_len, src_len
            )

            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        attn_output = sdpa_with_flattened_batch(q, k, v, attn_mask, dropout_p)


        attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    return attn_output