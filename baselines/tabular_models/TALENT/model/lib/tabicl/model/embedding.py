from __future__ import annotations

from typing import List, Optional
from collections import OrderedDict

import torch
from torch import nn, Tensor

from .layers import SkippableLinear
from .encoders import SetTransformer
from .inference import InferenceManager
from .inference_config import MgrConfig


class ColEmbedding(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        reserve_cls_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.reserve_cls_tokens = reserve_cls_tokens
        self.in_linear = SkippableLinear(1, embed_dim)

        self.tf_col = SetTransformer(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        self.out_w = SkippableLinear(embed_dim, embed_dim)
        self.ln_w = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()

        self.out_b = SkippableLinear(embed_dim, embed_dim)
        self.ln_b = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()

        self.inference_mgr = InferenceManager(enc_name="tf_col", out_dim=embed_dim)

    @staticmethod
    def map_feature_shuffle(reference_pattern: List[int], other_pattern: List[int]) -> List[int]:


        orig_to_other = {feature: idx for idx, feature in enumerate(other_pattern)}
        mapping = [orig_to_other[feature] for feature in reference_pattern]

        return mapping

    def _compute_embeddings(self, features: Tensor, train_size: Optional[int] = None) -> Tensor:


        src = self.in_linear(features)
        src = self.tf_col(src, train_size)
        weights = self.ln_w(self.out_w(src))
        biases = self.ln_b(self.out_b(src))
        embeddings = features * weights + biases

        return embeddings

    def _train_forward(self, X: Tensor, d: Optional[Tensor] = None, train_size: Optional[int] = None) -> Tensor:


        if self.reserve_cls_tokens > 0:

            X = nn.functional.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)

        if d is None:
            features = X.transpose(1, 2).unsqueeze(-1)
            embeddings = self._compute_embeddings(features, train_size)
        else:
            if self.reserve_cls_tokens > 0:
                d = d + self.reserve_cls_tokens

            B, T, HC = X.shape
            device = X.device
            X = X.transpose(1, 2)

            indices = torch.arange(HC, device=device).unsqueeze(0).expand(B, HC)
            mask = indices < d.unsqueeze(1)
            features = X[mask].unsqueeze(-1)
            effective_embeddings = self._compute_embeddings(features, train_size)

            embeddings = torch.zeros(B, HC, T, self.embed_dim, device=device)
            embeddings[mask] = effective_embeddings

        return embeddings.transpose(1, 2)

    def _inference_forward(
        self,
        X: Tensor,
        train_size: Optional[int] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        mgr_config: MgrConfig = None,
    ) -> Tensor:


        if mgr_config is None:
            mgr_config = MgrConfig(
                min_batch_size=1,
                safety_factor=0.8,
                offload="auto",
                auto_offload_pct=0.5,
                device=None,
                use_amp=True,
                verbose=False,
            )
        self.inference_mgr.configure(**mgr_config)

        if feature_shuffles is None:

            if self.reserve_cls_tokens > 0:

                X = nn.functional.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)

            features = X.transpose(1, 2).unsqueeze(-1)
            embeddings = self.inference_mgr(
                self._compute_embeddings, inputs=OrderedDict([("features", features), ("train_size", train_size)])
            )
        else:
            B = X.shape[0]

            first_table = X[0]
            if self.reserve_cls_tokens > 0:

                first_table = nn.functional.pad(first_table, (self.reserve_cls_tokens, 0), value=-100.0)

            features = first_table.transpose(0, 1).unsqueeze(-1)
            first_embeddings = self.inference_mgr(
                self._compute_embeddings,
                inputs=OrderedDict([("features", features), ("train_size", train_size)]),
                output_repeat=B,
            )


            embeddings = first_embeddings.unsqueeze(0).repeat(B, 1, 1, 1)
            first_pattern = feature_shuffles[0]
            for i in range(1, B):
                mapping = self.map_feature_shuffle(first_pattern, feature_shuffles[i])
                if self.reserve_cls_tokens > 0:
                    mapping = [m + self.reserve_cls_tokens for m in mapping]
                    mapping = list(range(self.reserve_cls_tokens)) + mapping
                embeddings[i] = first_embeddings[mapping]

        return embeddings.transpose(1, 2)

    def forward(
        self,
        X: Tensor,
        d: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        mgr_config: MgrConfig = None,
    ) -> Tensor:


        if self.training:
            embeddings = self._train_forward(X, d, train_size)
        else:
            embeddings = self._inference_forward(X, train_size, feature_shuffles, mgr_config)

        return embeddings