from __future__ import annotations

from collections import OrderedDict
import math
import torch
from torch import nn, Tensor

from .layers import ClassNode, OneHotAndLinear
from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import MgrConfig


class ICLearning(nn.Module):
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
        max_classes: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.max_classes = max_classes
        self.norm_first = norm_first

        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        if self.norm_first:
            self.ln = nn.LayerNorm(d_model)

        self.y_encoder = OneHotAndLinear(max_classes, d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, max_classes))

        self.inference_mgr = InferenceManager(enc_name="tf_icl", out_dim=max_classes)

    def _grouping(self, num_classes: int) -> tuple[Tensor, int]:
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
           

        if num_classes <= self.max_classes:
            return torch.zeros(num_classes, dtype=torch.int), 1

        num_groups = min(math.ceil(num_classes / self.max_classes), self.max_classes)
        group_assignments = torch.zeros(num_classes, dtype=torch.int)
        current_pos = 0

        remaining_classes = num_classes
        remaining_groups = num_groups
        for i in range(num_groups):
            group_size = math.ceil(remaining_classes / remaining_groups)
            group_assignments[current_pos : current_pos + group_size] = i
            current_pos += group_size
            remaining_classes -= group_size
            remaining_groups -= 1

        return group_assignments, num_groups

    def _fit_node(self, node: ClassNode, R: Tensor, y: Tensor, current_depth: int):
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
           

        unique_classes = torch.unique(y).int()
        node.classes_ = unique_classes

        if len(unique_classes) <= self.max_classes:
                                                        
            node.is_leaf = True
            node.R = R
            node.y = y
            return

                                   
        group_assignments, num_groups = self._grouping(len(unique_classes))

                                                                                        
        node.class_mapping = {c.item(): g.item() for c, g in zip(unique_classes, group_assignments)}
        node.group_indices = torch.tensor([node.class_mapping[c.item()] for c in y], dtype=torch.int)
        node.R = R
        node.y = y
        node.is_leaf = False

                                           
        for group in range(num_groups):
            mask = node.group_indices == group
            child_node = ClassNode(current_depth + 1)
            self._fit_node(child_node, R[mask], y[mask], current_depth + 1)
            node.child_nodes.append(child_node)

    def _fit_hierarchical(self, R_train: Tensor, y_train: Tensor):
\
\
\
\
\
\
\
\
\
           

        self.root = ClassNode(depth=0)
        self._fit_node(self.root, R_train, y_train, current_depth=0)

    def _label_encoding(self, y: Tensor) -> Tensor:
                                                                             

        unique_vals, _ = torch.unique(y, return_inverse=True)
        indices = unique_vals.argsort()
        return indices[torch.searchsorted(unique_vals, y)]

    def _icl_predictions(self, R: Tensor, y_train: Tensor) -> Tensor:
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
           

        train_size = y_train.shape[1]
        R[:, :train_size] = R[:, :train_size] + self.y_encoder(y_train.float())
        src = self.tf_icl(R, attn_mask=train_size)
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)                       

        return out

    def _predict_standard(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = False,
        softmax_temperature: float = 0.9,
        auto_batch: bool = True,
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
           

        train_size = y_train.shape[1]
        num_classes = len(torch.unique(y_train[0]))
        out = self.inference_mgr(
            self._icl_predictions, inputs=OrderedDict([("R", R), ("y_train", y_train)]), auto_batch=auto_batch
        )
        out = out[:, train_size:, :num_classes]

        if not return_logits:
            out = torch.softmax(out / softmax_temperature, dim=-1)

        return out

    def _predict_hierarchical(self, R_test: Tensor, softmax_temperature: float = 0.9) -> Tensor:
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
           

        test_size = R_test.shape[0]
        device = R_test.device
        num_classes = len(self.root.classes_)

        def process_node(node, R_test):
\
\
\
\
               

                                                  
            node_R = torch.cat([node.R.to(device), R_test], dim=0)

                                                       
            if node.is_leaf:
                node_y = self._label_encoding(node.y.to(device))
                                               
                leaf_preds = self._predict_standard(
                    R=node_R.unsqueeze(0),
                    y_train=node_y.unsqueeze(0),
                    softmax_temperature=softmax_temperature,
                    auto_batch=False,
                ).squeeze(0)
                                                                
                global_preds = torch.zeros((test_size, num_classes), device=device)
                for local_idx, global_idx in enumerate(node.classes_):
                    global_preds[:, global_idx] = leaf_preds[:, local_idx]

                return global_preds

                                                                
                                                      
            final_probs = torch.zeros((test_size, num_classes), device=device)

                                                   
            node_y = node.group_indices.to(device)
            group_probs = self._predict_standard(
                R=node_R.unsqueeze(0),
                y_train=node_y.unsqueeze(0),
                softmax_temperature=softmax_temperature,
                auto_batch=False,
            ).squeeze(0)

                                                                     
            for group_idx, child_node in enumerate(node.child_nodes):
                child_probs = process_node(child_node, R_test)
                final_probs += child_probs * group_probs[:, group_idx : group_idx + 1]

            return final_probs

        return process_node(self.root, R_test)

    def _inference_forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig = None,
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
           
                                        
        if mgr_config is None:
            mgr_config = MgrConfig(
                min_batch_size=1,
                safety_factor=0.8,
                offload=False,
                auto_offload_pct=0.5,
                device=None,
                use_amp=True,
                verbose=False,
            )
        self.inference_mgr.configure(**mgr_config)

        num_classes = len(torch.unique(y_train[0]))
        assert all(
            len(torch.unique(yi)) == num_classes for yi in y_train
        ), "All tables must have the same number of classes"

        if num_classes <= self.max_classes:
                                     
            out = self._predict_standard(
                R, y_train, return_logits=return_logits, softmax_temperature=softmax_temperature
            )
        else:
                                         
            out = []
            train_size = y_train.shape[1]
            for ri, yi in zip(R, y_train):
                if mgr_config.offload:
                    ri, yi = ri.cpu(), yi.cpu()
                else:
                    ri, yi = ri.to(mgr_config.device), yi.to(mgr_config.device)
                self._fit_hierarchical(ri[:train_size], yi)
                probs = self._predict_hierarchical(ri[train_size:])
                out.append(probs)
            out = torch.stack(out, dim=0)
            if return_logits:
                out = softmax_temperature * torch.log(out + 1e-6)

        return out

    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig = None,
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
           

        if self.training:
            train_size = y_train.shape[1]
            out = self._icl_predictions(R, y_train)
            out = out[:, train_size:]
        else:
            out = self._inference_forward(R, y_train, return_logits, softmax_temperature, mgr_config)

        return out