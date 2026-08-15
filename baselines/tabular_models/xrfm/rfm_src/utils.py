
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm, fractional_matrix_power

class SmoothClampedReLU(nn.Module):
    def __init__(self, beta=50):
        super(SmoothClampedReLU, self).__init__()
        self.beta = beta

    def forward(self, x):

        activated = F.softplus(x, beta=self.beta)


        clamped = activated - F.softplus(activated - 1, beta=self.beta)

        return clamped

def float_x(data):

    return np.float32(data)

def matrix_power(M, power):

    return stable_matrix_power(M, power)














def stable_matrix_power(M, power):

    if len(M.shape) == 2:
        assert M.shape[0] == M.shape[1], "Matrix must be square"
        if M.shape[0] < 700:
            M_cpu = M.cpu().float()
            M_cpu.diagonal().add_(1e-8)
            U, S, _ = torch.linalg.svd(M_cpu)
            S[S<0] = 0.
            return (U @ torch.diag(S**power) @ U.T).to(device=M.device, dtype=M.dtype)
        else:
            M.diagonal().add_(1e-8)
            S, U = torch.linalg.eigh(M)
            S[S<0] = 0.
            return (U @ torch.diag(S**power) @ U.T).to(device=M.device, dtype=M.dtype)

    elif len(M.shape) == 1:
        assert M.shape[0] > 0, "Vector must be non-empty"
        M[M<0] = 0.
        return M**power
    else:
        raise ValueError(f"Invalid matrix shape for square root: {M.shape}")
