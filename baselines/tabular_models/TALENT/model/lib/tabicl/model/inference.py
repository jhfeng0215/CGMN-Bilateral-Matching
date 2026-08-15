from __future__ import annotations

import warnings
import itertools
from collections import OrderedDict
from typing import List, Tuple, Dict, Callable, Iterator, Literal, Optional, Any

import psutil
from tqdm.auto import tqdm

import math
from scipy.optimize import fsolve

import torch
from torch import Tensor


class MemoryEstimator:



    coefficients: Dict[str, list] = {
        "tf_col": [7.079980260e-02, 7.29386080e-06, 3.90989142e-03],
        "tf_row": [-2.06831848e-05, 2.27205969e-04, 5.37117114e-03],
        "tf_icl": [-2.60068961e-01, 4.77470594e-07, 1.95310976e-02],
    }
    intercepts: Dict[str, float] = {
        "tf_col": 137.62474190864668,
        "tf_row": 138.53653545318957,
        "tf_icl": 140.58027172987750,
    }

    @staticmethod
    def estimate_peak_mem(
        batch_size: int, seq_len: int, enc_name: str, include_inputs: bool = True, in_dim: Optional[int] = None
    ) -> float:

        coefs = MemoryEstimator.coefficients[enc_name]
        inter = MemoryEstimator.intercepts[enc_name]
        peak_activation_mem = coefs[0] * batch_size + coefs[1] * seq_len + coefs[2] * batch_size * seq_len + inter

        if include_inputs:
            assert in_dim is not None, "Input dimension must be provided for input memory estimation"
            bytes_per_element = 4
            n_elements = batch_size * seq_len * in_dim
            mem_inputs = n_elements * bytes_per_element / (1024**2)
            peak_activation_mem += mem_inputs

        return peak_activation_mem

    @staticmethod
    def estimate_batch_size(
        seq_len: int, target_memory: float, enc_name: str, include_inputs: bool = True, in_dim: Optional[int] = None
    ) -> int:


        def objective_function(bs: float) -> float:
            return MemoryEstimator.estimate_peak_mem(bs, seq_len, enc_name, include_inputs, in_dim) - target_memory

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            solution = fsolve(objective_function, x0=1)[0]


        return max(1, int(solution))


class InferenceManager:


    def __init__(self, enc_name: str, out_dim: int, out_no_seq: bool = False):
        self.enc_name = enc_name
        self.out_dim = out_dim
        self.out_no_seq = out_no_seq
        self._is_configured = False

    def configure(
        self,
        min_batch_size: int = 1,
        safety_factor: float = 0.8,
        offload: bool | Literal["auto"] = "auto",
        auto_offload_pct: float = 0.5,
        device: Optional[str | torch.device] = None,
        use_amp: bool = True,
        verbose: bool = False,
    ):


        self.min_batch_size = min_batch_size
        self.safety_factor = safety_factor
        self.offload = offload
        self.auto_offload_pct = auto_offload_pct
        self.use_amp = use_amp
        self.verbose = verbose

        if device is None:
            self.exe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.exe_device = torch.device(device)
        else:
            self.exe_device = device

        self._is_configured = True

    def to_exe_device(self, tensor: Tensor) -> Tensor:

        if isinstance(tensor, torch.Tensor) and self.exe_device.type == "cuda" and not tensor.is_cuda:
            return tensor.to(self.exe_device)
        return tensor

    def get_available_cpu_memory(self) -> float:

        return psutil.virtual_memory().available / (1024 * 1024)

    def get_available_gpu_memory(self) -> float:

        if not torch.cuda.is_available() or self.exe_device.type != "cuda":
            return float("inf")


        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info(self.exe_device)[0] / (1024 * 1024)

    def estimate_safe_batch_size(
        self, seq_len: int, include_inputs: bool = True, in_dim: Optional[int] = None, max_bs: int = 50000
    ) -> Tuple[float, int]:

        available_mem = self.get_available_gpu_memory()
        target_mem = available_mem * self.safety_factor


        estimated_bs = MemoryEstimator.estimate_batch_size(seq_len, target_mem, self.enc_name, include_inputs, in_dim)


        if estimated_bs > max_bs and self.verbose:
            print(
                f"Warning: Estimated batch size {estimated_bs} exceeds maximum safe limit. "
                f"Capping batch size to {max_bs} to avoid CUDA configuration errors."
            )

        safe_bs = min(max(self.min_batch_size, estimated_bs), max_bs)

        return available_mem, safe_bs

    def __call__(
        self,
        forward_fn: Callable[..., Tensor],
        inputs: OrderedDict[str, Any],
        auto_batch: bool = True,
        output_repeat: int = 1,
    ) -> Tensor:


        if not hasattr(self, "_is_configured") or not self._is_configured:
            raise RuntimeError(
                "InferenceManager must be configured before running inference. Call configure_inference() first."
            )

        if not auto_batch:

            inputs_on_exe = {}
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs_on_exe[name] = self.to_exe_device(value)
                else:
                    inputs_on_exe[name] = value

            with torch.no_grad():
                if self.use_amp and self.exe_device.type == "cuda":
                    with torch.autocast(device_type="cuda"):
                        outputs = forward_fn(**inputs_on_exe)
                else:
                    outputs = forward_fn(**inputs_on_exe)


            if self.offload:
                return outputs.to(device="cpu")
            else:
                return outputs


        if self.exe_device.type == "cpu":
            return forward_fn(**inputs)


        first_value = next(iter(inputs.values()))

        if not isinstance(first_value, torch.Tensor):
            raise ValueError("First input must be a tensor.")

        if first_value.dim() < 3:
            raise ValueError(
                f"First tensor input must have at least 3 dimensions (batch_dim, seq_len, in_dim), "
                f"got {first_value.dim()}"
            )


        *batch_dims, seq_len, in_dim = first_value.shape
        input_dtype = first_value.dtype
        inputs_on_cuda = first_value.is_cuda


        total_bs = math.prod(batch_dims)


        gpu_mem, batch_size = self.estimate_safe_batch_size(seq_len, include_inputs=not inputs_on_cuda, in_dim=in_dim)

        if self.verbose:
            print(
                f"\nAvailable memory: {gpu_mem / 1024:.2f}GB, sequence length: {seq_len}, "
                f"estimated batch elements per batch for {self.enc_name}: {batch_size}\n"
            )


        if self.out_no_seq:
            output_shape = (*batch_dims, self.out_dim)
        else:
            output_shape = (*batch_dims, seq_len, self.out_dim)


        if self.offload == "auto":

            bytes_per_element = torch.tensor([], dtype=input_dtype).element_size()
            output_mb = bytes_per_element * math.prod(output_shape) / (1024 * 1024)
            output_mb *= output_repeat


            output_pct = output_mb / gpu_mem
            excess_gpu = output_pct > self.auto_offload_pct


            cpu_mem = self.get_available_cpu_memory()
            enough_cpu = cpu_mem > output_mb

            self.offload = excess_gpu and enough_cpu
            if self.verbose:
                print(
                    f"Output size: {output_mb / 1024:.2f}GB, "
                    f"CPU memory: {cpu_mem / 1024:.2f}GB, "
                    f"GPU memory: {gpu_mem / 1024:.2f}GB\n"
                    f"Output size exceeds {self.auto_offload_pct * 100:.2f}% of GPU memory: {excess_gpu} "
                    f"and CPU memory is sufficient: {enough_cpu}\n"
                    f"Offloading to CPU: {self.offload}"
                )


        if batch_size >= total_bs:

            inputs_on_exe = {}
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs_on_exe[name] = self.to_exe_device(value)
                else:
                    inputs_on_exe[name] = value

            with torch.no_grad():
                if self.use_amp and self.exe_device.type == "cuda":
                    with torch.autocast(device_type="cuda"):
                        outputs = forward_fn(**inputs_on_exe)
                else:
                    outputs = forward_fn(**inputs_on_exe)


            if self.offload:
                return outputs.to(device="cpu")
            else:
                return outputs


        output_device = torch.device("cpu") if self.offload else self.exe_device
        outputs = torch.empty(output_shape, dtype=input_dtype, device=output_device)


        while True:
            try:

                split_sizes = self.compute_split_sizes(batch_dims, batch_size)

                n_batches = self.compute_n_batches(batch_dims, split_sizes)
                batch_iterator = self.create_multidim_batches(inputs, batch_dims, split_sizes)

                if self.verbose:
                    batch_iterator = tqdm(
                        batch_iterator, total=n_batches, desc=f"Processing {self.enc_name}", unit="batch"
                    )

                for batch_dict, indices in batch_iterator:
                    with torch.no_grad():
                        if self.use_amp and self.exe_device.type == "cuda":
                            with torch.autocast(device_type="cuda"):
                                output = forward_fn(**batch_dict)
                        else:
                            output = forward_fn(**batch_dict)

                        if self.offload:

                            outputs[indices] = output.to(device="cpu")
                            del output
                        else:
                            outputs[indices] = output


                    del batch_dict

                return outputs

            except torch.cuda.OutOfMemoryError as e:
                if batch_size <= self.min_batch_size:
                    raise RuntimeError(
                        f"Failed to execute even with minimum batch size {self.min_batch_size}. Error: {e}"
                    )

                if self.verbose:
                    print(
                        f"OOM with batch_size={batch_size} for {self.enc_name}, "
                        f"reducing to {max(self.min_batch_size, batch_size // 2)}"
                    )


                if self.exe_device.type == "cuda":
                    torch.cuda.empty_cache()

                batch_size = max(self.min_batch_size, batch_size // 2)

    @staticmethod
    def compute_split_sizes(batch_dims: Tuple[int], batch_size: int) -> List[int]:

        if not batch_dims:
            return []


        elements_left = batch_size
        split_sizes = []


        for dim_size in batch_dims:
            if elements_left >= dim_size:

                split_sizes.append(dim_size)
                elements_left //= dim_size
            else:

                split_sizes.append(min(dim_size, max(1, elements_left)))
                elements_left = 0
                break


        split_sizes.extend([1] * (len(batch_dims) - len(split_sizes)))

        return split_sizes

    @staticmethod
    def compute_n_batches(batch_dims: Tuple[int], split_sizes: List[int]) -> int:


        n_batches = 1
        for batch_dim, split_size in zip(batch_dims, split_sizes):
            n_batches *= math.ceil(batch_dim / split_size)

        return n_batches

    def create_multidim_batches(
        self, inputs: OrderedDict[str, Any], batch_dims: Tuple[int], split_sizes: List[int]
    ) -> Iterator:



        slices = []
        for dim_size, batch_size in zip(batch_dims, split_sizes):
            dim_slices = []
            for start in range(0, dim_size, batch_size):
                end = min(start + batch_size, dim_size)
                dim_slices.append(slice(start, end))
            slices.append(dim_slices)


        slice_tuples = itertools.product(*slices)

        for slice_tuple in slice_tuples:
            batch_dict = {}
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):

                    batch_dict[name] = self.to_exe_device(value[slice_tuple])
                else:

                    batch_dict[name] = value

            yield batch_dict, slice_tuple