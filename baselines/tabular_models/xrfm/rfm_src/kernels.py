from typing import Optional, Union

import torch
from tqdm import tqdm
import time
from typing import List

import torch.nn.functional as F

def get_sub_matrix(mat: Union[torch.Tensor, None], indices: torch.Tensor) -> Union[torch.Tensor, None]:

    if mat is None:
        return None
    if len(mat.shape) == 1:
        return mat[indices]
    else:
        return mat[indices][:, indices]

class Kernel:
    def __init__(self):
        self.is_adaptive_bandwidth = True
        self.handle_categorical = False
        self.use_sqrtM = True

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _transform_m(self, x: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:

        if mat is not None:
            if len(mat.shape) == 1:

                x = x * mat[None, :].to(dtype=x.dtype)
            elif len(mat.shape) == 2:
                x = x @ mat.to(dtype=x.dtype)
            else:
                raise ValueError(f'm_matrix should have one or two dimensions, but got shape {mat.shape}')
        return x

    def _reset_adaptive_bandwidth(self):
        self.is_adaptive_bandwidth = False
        return

    def _adapt_bandwidth(self, kernel_mat: torch.Tensor, adapt_mode='median', sub_mat_size=5000):
        n, m = kernel_mat.shape
        assert n <= m, "Kernel matrix must be wider than it is tall"


        sub_mat_size = min(sub_mat_size, n)
        sample_indices = torch.randperm(n)[:sub_mat_size]
        sample_matrix = kernel_mat[sample_indices][:, sample_indices]


        mask = ~torch.eye(sub_mat_size, dtype=bool, device=kernel_mat.device)


        if adapt_mode == 'median':
            bandwidth_multiplier = torch.median(sample_matrix[mask])
        elif adapt_mode == 'mean':
            bandwidth_multiplier = torch.mean(sample_matrix[mask])
        else:
            raise ValueError(f"Invalid adapt_mode: {adapt_mode}")
        self.bandwidth = self.base_bandwidth * bandwidth_multiplier.item()
        self.is_adaptive_bandwidth = True
        return


    def set_categorical_indices(self, numerical_indices, categorical_indices, categorical_vectors, device='cuda'):
        print("Setting categorical indices")
        self.numerical_indices = numerical_indices.to(device)
        self.categorical_indices = [categorical_indices[i].to(device) for i in range(len(categorical_indices))]
        self.categorical_vectors = [categorical_vectors[i].to(device) for i in range(len(categorical_vectors))]
        self.handle_categorical = True
        return

    def get_agop(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                 mat: Optional[torch.Tensor] = None, center_grads: bool = False) -> torch.Tensor:
        if self.handle_categorical:
            return self.get_agop_categorical(x, z, coefs, mat, center_grads)
        else:

            f_grads = self.get_function_grads(x, z, coefs, mat)

            f_grads = f_grads.reshape(-1, f_grads.shape[-1])
            if center_grads:
                f_grads = f_grads - f_grads.mean(dim=0, keepdim=True)
            return f_grads.transpose(-1, -2) @ f_grads

    def get_agop_categorical(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                             mat: Optional[torch.Tensor] = None, center_grads: bool = False) -> torch.Tensor:
        raise NotImplementedError()

    def get_kernel_matrix(self, x: torch.Tensor, z: torch.Tensor,
                          mat: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.handle_categorical:
            return self._get_kernel_matrix_categorical_impl(x, z, mat)
        else:
            return self._get_kernel_matrix_impl(x, z, mat)

    def _get_kernel_matrix_categorical_impl(self, x: torch.Tensor, z: torch.Tensor,
                          mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    def get_kernel_matrix_symm(self, x: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:

        return self.get_kernel_matrix(x, x, mat)

    def get_function_grads(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                           mat: Optional[torch.Tensor] = None) -> torch.Tensor:

        grads = self._get_function_grad_impl(self._transform_m(x, mat), self._transform_m(z, mat), coefs)
        return self._transform_m(grads, mat)

    def get_agop_diag(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                      mat: Optional[torch.Tensor] = None, center_grads: bool = False) -> torch.Tensor:

        f_grads = self.get_function_grads(x, z, coefs, mat)

        f_grads = f_grads.reshape(-1, f_grads.shape[-1])
        if center_grads:
            f_grads = f_grads - f_grads.mean(dim=0, keepdim=True)
        return f_grads.square().sum(dim=-2)

class LightLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert exponent > 0
        assert eps > 0
        self.bandwidth_mode = bandwidth_mode
        self.base_bandwidth = bandwidth
        self.bandwidth = bandwidth
        self.exponent = exponent
        self.eps = eps
        self.use_sqrtM = False

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        xm = self._transform_m(x, mat)
        zm = self._transform_m(z, mat)

        xm_norm_sqr = (xm*x).sum(dim=-1)
        zm_norm_sqr = (zm*z).sum(dim=-1)

        kernel_mat = xm_norm_sqr[:, None] - 2*xm@z.T + zm_norm_sqr[None, :]
        kernel_mat.clamp_(min=0)
        kernel_mat.sqrt_()

        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)
        if self.exponent != 1.0:
            kernel_mat.pow_(self.exponent)

        kernel_mat.mul_(-1. / (self.bandwidth ** self.exponent))
        kernel_mat.exp_()
        return kernel_mat

    def get_function_grads(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                           mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        xm = self._transform_m(x, mat)
        zm = self._transform_m(z, mat)

        xm_norm_sqr = (xm*x).sum(dim=-1)
        zm_norm_sqr = (zm*z).sum(dim=-1)

        dists = xm_norm_sqr[:, None] - 2*xm@z.T + zm_norm_sqr[None, :]
        dists.clamp_(min=0)
        dists.sqrt_()

        gamma = 1. / self.bandwidth
        kernel_mat = dists ** self.exponent
        kernel_mat.mul_(-gamma)
        kernel_mat.exp_()


        mask = dists >= self.eps
        dists.clamp_(min=self.eps)
        dists.pow_(self.exponent - 2)
        kernel_mat.mul_(dists)
        kernel_mat.mul_(mask)
        kernel_mat.mul_(-gamma * self.exponent)


        return torch.einsum('li,ij,jd->ljd', coefs, kernel_mat, zm) - torch.einsum('li,ij,id->ljd', coefs, kernel_mat, xm)

class LaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert exponent > 0
        assert eps > 0
        self.bandwidth_mode = bandwidth_mode
        self.base_bandwidth = bandwidth
        self.bandwidth = bandwidth
        self.exponent = exponent
        self.eps = eps

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        kernel_mat = torch.cdist(self._transform_m(x, mat), self._transform_m(z, mat))
        kernel_mat.clamp_(min=0)
        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)
        if self.exponent != 1.0:
            kernel_mat.pow_(self.exponent)

        kernel_mat.mul_(-1. / (self.bandwidth ** self.exponent))
        kernel_mat.exp_()
        return kernel_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, z)
        dists.clamp_(min=0)






        gamma = 1. / self.bandwidth
        kernel_mat = dists ** self.exponent
        kernel_mat.mul_(-gamma)
        kernel_mat.exp_()


        mask = dists >= self.eps
        dists.clamp_(min=self.eps)
        dists.pow_(self.exponent - 2)
        kernel_mat.mul_(dists)
        kernel_mat.mul_(mask)
        kernel_mat.mul_(-gamma * self.exponent)


        return torch.einsum('li,ij,jd->ljd', coefs, kernel_mat, z) - torch.einsum('li,ij,id->ljd', coefs, kernel_mat, x)

class ProductLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert exponent > 0
        assert eps > 0
        self.bandwidth_mode = bandwidth_mode
        self.base_bandwidth = bandwidth
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.exponent = exponent
        self.eps = eps

    def get_sample_batch_size(self, n: int, d: int, scalar_size: int = 4, mem_constant: float = 20) -> int:
        total_memory_possible = torch.cuda.get_device_properties(torch.device('cuda')).total_memory
        curr_mem_use = torch.cuda.memory_allocated()
        available_memory = total_memory_possible - curr_mem_use
        return int(available_memory / (mem_constant*n*scalar_size))

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None, kernel_batch_size=20_000) -> torch.Tensor:
        n, d = z.shape
        if x.shape[0] <= kernel_batch_size:
            kernel_mat = torch.cdist(self._transform_m(x, mat), self._transform_m(z, mat), p=self.exponent)
        else:
            kernel_mat = torch.empty((x.shape[0], z.shape[0]), device=x.device, dtype=x.dtype)
            for i in range(0, x.shape[0], kernel_batch_size):
                kernel_mat[i:i+kernel_batch_size] = torch.cdist(self._transform_m(x[i:i+kernel_batch_size], mat), self._transform_m(z, mat), p=self.exponent)

        kernel_mat.clamp_(min=0)
        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)
        kernel_mat.pow_(self.exponent)
        kernel_mat.mul_(-1./(self.bandwidth**self.exponent))
        kernel_mat.exp_()
        return kernel_mat

    def _get_kernel_matrix_categorical_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        numerical_indices = self.numerical_indices
        categorical_indices = self.categorical_indices
        categorical_vectors = self.categorical_vectors

        assert len(numerical_indices) > 0 or len(categorical_indices) > 0, "No numerical or categorical features"
        assert len(categorical_indices) == len(categorical_vectors), "Number of categorical index and vector groups must match"

        def dist_fn(x, z):
            kernel_mat = torch.cdist(x, z, p=self.exponent)
            kernel_mat.clamp_(min=0)
            if not self.is_adaptive_bandwidth:
                self._adapt_bandwidth(kernel_mat)
            if self.exponent != 1.0:
                kernel_mat.pow_(self.exponent)
            return kernel_mat


        mat_num = get_sub_matrix(mat, numerical_indices)
        xnum = self._transform_m(x[:, numerical_indices], mat_num)
        znum = self._transform_m(z[:, numerical_indices], mat_num)

        batch_size = self.get_sample_batch_size(znum.shape[0], znum.shape[1])
        dist_mat = torch.zeros((xnum.shape[0], znum.shape[0]), device=xnum.device, dtype=xnum.dtype)
        num_batch_size = 2*batch_size

        for i in range(0, xnum.shape[0], num_batch_size):
            dist_mat[i:i+num_batch_size, :] = dist_fn(xnum[i:i+num_batch_size], znum)
        for cat_idx, cat_vecs in zip(categorical_indices, categorical_vectors):

            x_cat = x[:, cat_idx].argmax(dim=-1)
            z_cat = z[:, cat_idx].argmax(dim=-1)


            mat_cat = get_sub_matrix(mat, cat_idx)
            cat_vecs_transformed = self._transform_m(cat_vecs, mat_cat)
            cat_embedding_kernel = dist_fn(cat_vecs_transformed, cat_vecs_transformed)


            for i in range(0, x.shape[0], batch_size):
                dist_mat[i:i+batch_size].add_(cat_embedding_kernel[x_cat[i:i+batch_size, None], z_cat[None, :]])

        dist_mat.mul_(-1./(self.bandwidth**self.exponent))
        dist_mat.exp_()
        return dist_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        def forward_func(z):
            dists = torch.cdist(x, z, p=self.exponent) ** self.exponent
            factor = -((1. / self.bandwidth) ** self.exponent)

            return coefs @ torch.exp(factor * (dists * (dists >= self.eps))).sum(dim=1)

        return torch.func.jacrev(forward_func)(z)

    def get_agop_categorical(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                 mat: Optional[torch.Tensor] = None, center_grads: bool = False) -> torch.Tensor:

        numerical_indices = self.numerical_indices
        categorical_indices = self.categorical_indices


        f_grads = self.get_function_grads(x, z, coefs, mat)

        f_grads = f_grads.reshape(-1, f_grads.shape[-1])
        if center_grads:
            f_grads = f_grads - f_grads.mean(dim=0, keepdim=True)


        d = x.shape[1]
        agop = torch.zeros((d, d), device=x.device, dtype=x.dtype)


        if numerical_indices is not None and len(numerical_indices) > 0:
            agop[numerical_indices[:, None], numerical_indices] = (
                f_grads[:, numerical_indices].T @ f_grads[:, numerical_indices]
            )


        if categorical_indices is not None:
            for cat_idx in categorical_indices:
                agop[cat_idx[:, None], cat_idx] = (
                    f_grads[:, cat_idx].T @ f_grads[:, cat_idx]
                )

        return agop


class LpqLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, p: float, q: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert 0 < p <= 2
        assert 0 < q <= p
        assert eps > 0
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.p = p
        self.q = q
        self.eps = eps
        self.bandwidth_mode = bandwidth_mode

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.p == 2:

            kernel = LaplaceKernel(bandwidth=self.bandwidth, exponent=self.q, eps=self.eps, bandwidth_mode=self.bandwidth_mode)
            return kernel.get_kernel_matrix(x, z)

        kernel_mat = torch.cdist(x, z, p=self.p)
        kernel_mat.clamp_(min=0)
        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)
        kernel_mat.pow_(self.q)
        kernel_mat.mul_(-1. / (self.bandwidth ** self.q))
        kernel_mat.exp_()
        return kernel_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        if self.p == 2:

            kernel = LaplaceKernel(bandwidth=self.bandwidth, exponent=self.q, eps=self.eps,
                                   bandwidth_mode=self.bandwidth_mode)
            return kernel.get_function_grads(x, z, coefs)

        def forward_func(z):
            dists = torch.cdist(x, z, p=self.p) ** self.q
            factor = -((1. / self.bandwidth) ** self.q)

            return coefs @ torch.exp(factor * (dists * (dists >= self.eps))).sum(dim=1)

        return torch.func.jacrev(forward_func)(z)

class SumPowerLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-10, const_mix: float = 0.0,
                 power: int = 2,
                 bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert exponent > 0
        assert eps > 0
        assert 0 <= const_mix < 1
        assert bandwidth_mode == 'constant', 'Adaptive bandwidth currently not supported'
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.exponent = exponent
        self.const_mix = const_mix
        self.power = power
        self.eps = eps
        self.bandwidth_mode = bandwidth_mode

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        diffs = x[:, None, :] - z[None, :, :]
        diffs.abs_()
        diffs.pow_(self.exponent)
        diffs.mul_(-1. / (self.bandwidth ** self.exponent))
        diffs.exp_()
        sum = diffs.sum(dim=-1)
        sum.mul_((1.0 - self.const_mix) / x.shape[1])
        sum.add_(self.const_mix)
        sum.pow_(self.power)
        return sum

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        def forward_func(z):

            diffs = torch.abs(x[:, None, :] - z[None, :, :]).pow(self.exponent)
            diffs = torch.exp((-1. / (self.bandwidth ** self.exponent)) * diffs)
            sum = (1.0 - self.const_mix) * (diffs.sum(dim=-1) / x.shape[-1]) + self.const_mix
            sum = sum ** self.power
            sum = sum.sum(dim=-1)
            return coefs @ sum

        return torch.func.jacrev(forward_func)(z)


if __name__ == '__main__':

    kernel = ProductLaplaceKernel(bandwidth=2.0, exponent=1.2)

    n_samples = 2000
    n_features = 100
    x = torch.rand(n_samples, n_features)
    coefs = torch.rand(1, n_samples)
    kernel.get_agop(x, x, coefs)

    print('here')

    import matplotlib.pyplot as plt

    x = torch.linspace(-2.0, 2.0, 5)[:, None]
    z = torch.linspace(-4.0, 4.0, 500)[:, None]
    coefs = torch.as_tensor([[1.0, 0.8, 0.4, -0.5, -2.0], [0.1, 0.2, 0.3, 0.4, 0.5]])

    mat = torch.as_tensor([0.5])

    f_values = coefs[0, :] @ kernel.get_kernel_matrix(x, z, mat)
    plt.plot(z[:, 0], f_values, 'tab:blue', label='function')
    plt.plot(z, kernel.get_function_grads(x, z, coefs, mat)[0], 'tab:orange', label='gradient')
    plt.plot(0.5 * (z[1:, 0] + z[:-1, 0]), (f_values[1:] - f_values[:-1]) / (z[1:, 0] - z[:-1, 0]), color='tab:green',
             linestyle='--', label='finite diff')
    plt.legend()
    plt.show()
