from __future__ import annotations
from typing import List, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .rope import RotaryEmbedding
from .attention import multi_head_attention_forward


class ClassNode:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = False
        self.classes_ = None
        self.child_nodes = []
        self.class_mapping = {}
        self.group_indices = None
        self.R = None
        self.y = None


class OneHotAndLinear(nn.Linear):
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes, embed_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def forward(self, src: Tensor) -> Tensor:
\
\
\
\
\
\
\
\
\
\
\
           
                                                                        
        one_hot = F.one_hot(src.long(), self.num_classes).to(src.dtype)
        return F.linear(one_hot, self.weight, self.bias)


class SkippableLinear(nn.Linear):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, in_features: int, out_features: int, bias: bool = True, skip_value: float = -100.0):
        super().__init__(in_features, out_features, bias)
        self.skip_value = skip_value

    def forward(self, src: Tensor) -> Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
           

        out = F.linear(src, self.weight, self.bias)
        skip_mask = (src == self.skip_value).all(dim=-1)
        if skip_mask.any():
            out[skip_mask] = self.skip_value

        return out


class MLP(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
                                    
        act_fn = self.get_activation(activation)
        layers = []

                                               
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            layers.append(act_fn())
            prev_dim = hidden_dim

                                    
        if out_dim is not None:
            layers.append(nn.Linear(prev_dim, out_dim, bias=bias))

        self.net = nn.Sequential(*layers)

    @staticmethod
    def get_activation(activation: str) -> nn.Module:
\
\
\
\
\
\
\
\
\
\
\
           

        activation_map = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }

        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}. Supported: {list(activation_map.keys())}")

        return activation_map[activation]

    def forward(self, X: Tensor) -> Tensor:
\
\
\
\
\
\
\
\
\
\
\
           
        return self.net(X)


class MultiheadAttention(nn.MultiheadAttention):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout, batch_first=True)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

        if isinstance(attn_mask, int):
            assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
            assert rope is None, "Rotary position embedding is not supported with attn_mask as int"

        return multi_head_attention_forward(
            query,
            key,
            value,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
        )


class MultiheadAttentionBlock(nn.TransformerEncoderLayer):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, norm_first=norm_first, batch_first=True)
        del self.self_attn
        self.attn = MultiheadAttention(d_model, nhead, dropout)
        self.init_weights()

    def init_weights(self):
                                                                       
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

        if isinstance(attn_mask, int):
            assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
            assert rope is None, "Rotary position embedding is not supported with attn_mask as int"
        else:
                                                              
            key_padding_mask = F._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=F._none_or_dtype(attn_mask),
                other_name="src_mask",
                target_type=q.dtype,
            )
            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=q.dtype,
                check_other=False,
            )

                                      
        k = q if k is None else k
        v = q if v is None else v

                                                      
        x = q
        if self.norm_first:
                                                          
            attn = self._attn_block(self.norm1(q), self.norm1(k), self.norm1(v), key_padding_mask, attn_mask, rope)
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
                                                          
            attn = self._attn_block(q, k, v, key_padding_mask, attn_mask, rope)
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))

        return x

    def _attn_block(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor | int],
        rope: Optional[RotaryEmbedding],
    ) -> Tensor:
        attn = self.attn(q, k, v, key_padding_mask, attn_mask, rope)
        return self.dropout1(attn)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class InducedSelfAttentionBlock(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        skip_value: float = -100.0,
    ):
        super().__init__()
        self.skip_value = skip_value

                                       
        self.multihead_attn1 = MultiheadAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)
        self.multihead_attn2 = MultiheadAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)

                                   
        self.num_inds = num_inds
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

    def induced_attention(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

        *batch_shape, _, d_model = src.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)

        if train_size is None:
            hidden = self.multihead_attn1(ind_vectors, src, src)
        else:
            hidden = self.multihead_attn1(ind_vectors, src[..., :train_size, :], src[..., :train_size, :])

        out = self.multihead_attn2(src, hidden, hidden)

        return out

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

                                                                               
                             
                                 
                                                             
                   
                                             
                                                                                       
                                                  
               
                                                           

                    
        out = self.induced_attention(src, train_size)
        return out