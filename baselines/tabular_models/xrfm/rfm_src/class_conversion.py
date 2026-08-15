from typing import Literal, Optional
import torch
import torch.nn.functional as F

class ClassificationConverter:
    def __init__(self, mode: Literal['zero_one', 'prevalence'], n_classes: int, labels: Optional[torch.Tensor] = None):

        assert mode in ['zero_one', 'prevalence']
        assert n_classes >= 2
        self.mode = mode
        self.n_classes = n_classes

        if self.mode == 'prevalence':
            if labels is None:
                raise ValueError("labels must be provided for mode='prevalence'.")
            counts = torch.bincount(labels.cpu().long(), minlength=n_classes).float()
            total = counts.sum()
            if total.item() == 0:
                raise ValueError("labels must contain at least one element for mode='prevalence'.")
            prior = counts / total
            K = n_classes

            I = torch.eye(K, dtype=torch.float32)
            M = I[:, :-1] - I[:, [-1]]
            Q, _ = torch.linalg.qr(M, mode='reduced')

            mu = prior @ Q
            C = Q - mu

            A = torch.cat([C.T, torch.ones(1, K, dtype=torch.float32)], dim=0)
            invA = torch.linalg.inv(A)

            self._prior = prior
            self._C = C
            self._invA = invA
        else:
            self._prior = None
            self._C = None
            self._invA = None


    def labels_to_numerical(self, labels: torch.Tensor) -> torch.Tensor:

        if self.mode == 'zero_one':
            if self.n_classes == 2:
                return labels.float().unsqueeze(-1)
            return F.one_hot(labels, num_classes=self.n_classes).float()


        C = self._C.to(labels.device)
        return C[labels.long()]

    def numerical_to_probas(self, num: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:

        if self.mode == 'zero_one':
            if num.ndim == 1:
                num = num.unsqueeze(-1)
            if num.shape[1] == 1:
                num = torch.cat([1 - num, num], dim=1)
            num = torch.clamp(num, eps, 1 - eps)
            return num / num.sum(dim=1, keepdim=True)


        if num.ndim == 1:
            num = num.unsqueeze(0)
        invA = self._invA.to(num.device)
        N = num.shape[0]
        ones = torch.ones((N, 1), dtype=invA.dtype, device=num.device)
        B = torch.cat([num.to(dtype=invA.dtype), ones], dim=1)
        pi = B @ invA.T
        pi = torch.clamp(pi, eps, 1 - eps)
        return pi / pi.sum(dim=1, keepdim=True)

    def numerical_to_labels(self, num: torch.Tensor) -> torch.Tensor:

        probs = self.numerical_to_probas(num)
        return probs.argmax(dim=-1)
