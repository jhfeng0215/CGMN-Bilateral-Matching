from .class_conversion import ClassificationConverter
from .eigenpro import KernelModel
    
import torch, numpy as np
from .kernels import Kernel, LaplaceKernel, ProductLaplaceKernel, SumPowerLaplaceKernel, LightLaplaceKernel
from tqdm.contrib import tenumerate

from .metrics import Metrics, Metric
from .utils import matrix_power, SmoothClampedReLU
from .gpu_utils import with_env_var
from sklearn.metrics import roc_auc_score
import time
from typing import Union

class RFM(torch.nn.Module):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, kernel: Union[Kernel, str], iters=5, bandwidth=10., exponent=1., bandwidth_mode='constant', 
                 agop_power=0.5, device=None, diag=False, verbose=True, mem_gb=None, tuning_metric='mse', 
                 categorical_info=None, fast_categorical=True, class_converter=None, time_limit_s=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        super().__init__()
        if isinstance(kernel, str):
            kernel = self.kernel_from_str(kernel, bandwidth=bandwidth, exponent=exponent)
        self.kernel_obj = kernel
        self.agop_power = agop_power
        self.M = None
        self.sqrtM = None
        self.iters = iters
        self.diag = diag                                                 
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agop_power = 0.5                         
        self.max_lstsq_size = 70_000                                               
        self.bandwidth_mode = bandwidth_mode
        self.proba_beta = 500
        self.verbose = verbose
        self.tuning_metric = tuning_metric
        self.use_sqrtM = self.kernel_obj.use_sqrtM
        self.class_converter = class_converter
        self.time_limit_s = time_limit_s
        
        if categorical_info is not None and fast_categorical: 
            if isinstance(self.kernel_obj, ProductLaplaceKernel):
                self.set_categorical_indices(**categorical_info)
            else:
                print("Ignoring categorical indices for non-ProductLaplaceKernel.")

        if mem_gb is not None:
            self.mem_gb = mem_gb
        elif torch.cuda.is_available():
                                                                 
            self.mem_gb = torch.cuda.get_device_properties(self.device).total_memory//1024**3 - 1 
        else:
            self.mem_gb = 8
        
    def kernel(self, x, z):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        return self.kernel_obj.get_kernel_matrix(x, z, self.sqrtM if self.use_sqrtM else self.M)

    def kernel_from_str(self, kernel_str, bandwidth, exponent):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        if kernel_str in ['laplace', 'l2']:
            return LaplaceKernel(bandwidth=bandwidth, exponent=exponent)
        elif kernel_str in ['l2_high_dim', 'l2_light']:
            return LightLaplaceKernel(bandwidth=bandwidth, exponent=exponent)
        elif kernel_str in ['product_laplace', 'l1']:
            return ProductLaplaceKernel(bandwidth=bandwidth, exponent=exponent)
        elif kernel_str in ['sum_power_laplace', 'l1_power']:
            return SumPowerLaplaceKernel(bandwidth=bandwidth, exponent=exponent)
        else:
            raise ValueError(f"Invalid kernel: {kernel_str}")
        
    def update_M(self, samples):
\
\
\
\
\
\
\
\
\
\
\
\
           
        samples = samples.to(self.device)
        self.centers = self.centers.to(self.device)

        if self.M is None:
            if self.diag:
                self.M = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            else:
                self.M = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        if self.use_sqrtM and self.sqrtM is None:
            if self.diag:
                self.sqrtM = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            else:
                self.sqrtM = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        agop_func = self.kernel_obj.get_agop_diag if self.diag else self.kernel_obj.get_agop
        agop = agop_func(x=self.centers, z=samples, coefs=self.weights.t(), mat=self.sqrtM if self.use_sqrtM else self.M, center_grads=self.center_grads)
        return agop
    
    def reset_adaptive_bandwidth(self):
\
\
\
\
\
           
        self.kernel_obj._reset_adaptive_bandwidth()
        return 

    def tensor_copy(self, tensor):
\
\
\
\
\
\
\
\
\
\
\
\
           
        if tensor is None:
            return None
        elif self.keep_device or tensor.device.type == 'cpu':
            return tensor.clone()
        else:
            return tensor.cpu()
        
    def set_categorical_indices(self, numerical_indices, categorical_indices, categorical_vectors, device=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        if numerical_indices is None and categorical_indices is None and categorical_vectors is None:
            if self.verbose:
                print("No categorical indices provided, ignoring")
            return
        assert numerical_indices is not None, "Numerical indices must be provided if one of categorical indices/vectors are provided"
        assert categorical_vectors is not None, "Categorical vectors must be provided if categorical indices are provided"
        assert len(categorical_indices) == len(categorical_vectors), "Number of categorical index and vector groups must match"
        assert len(numerical_indices) > 0 or len(categorical_indices) > 0, "No numerical or categorical features"
        self.kernel_obj.set_categorical_indices(numerical_indices, categorical_indices, categorical_vectors, device=self.device if device is None else device)
        return

    def update_best_params(self, best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth, current_metric, current_iter):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
                                                                               
                                                                                                                                                           
        maximize_metric = Metric.from_name(self.tuning_metric).should_maximize
        if maximize_metric and current_metric > best_metric:
            best_metric = current_metric
            best_alphas = self.tensor_copy(self.weights)
            best_iter = current_iter
            best_bandwidth = self.kernel_obj.bandwidth+0
            best_M = self.tensor_copy(self.M)
            best_sqrtM = self.tensor_copy(self.sqrtM)

        elif not maximize_metric and current_metric < best_metric:
            best_metric = current_metric
            best_alphas = self.tensor_copy(self.weights)
            best_iter = current_iter
            best_bandwidth = self.kernel_obj.bandwidth+0
            best_M = self.tensor_copy(self.M)
            best_sqrtM = self.tensor_copy(self.sqrtM)

        return best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth
        
    def fit_predictor(self, centers, targets, bs=None, lr_scale=1, solver='solve', **kwargs):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        
        if self.bandwidth_mode == 'adaptive':
                                                                         
            print("Resetting adaptive bandwidth")
            self.reset_adaptive_bandwidth()

        self.centers = centers

        if self.fit_using_eigenpro:
            assert not self.label_centering, "EigenPro does not yet support label centering"
            if self.prefit_eigenpro:
                random_indices = torch.randperm(centers.shape[0])[:self.max_lstsq_size]
                if self.verbose:
                    print(f"Prefitting Eigenpro with {len(random_indices)} points")
                sub_weights = self.fit_predictor_lstsq(centers[random_indices], targets[random_indices], solver=solver)
                initial_weights = torch.zeros_like(targets)
                initial_weights[random_indices] = sub_weights.to(targets.device, dtype=targets.dtype)
            else:
                initial_weights = None

            self.weights = self.fit_predictor_eigenpro(centers, targets, bs=bs, lr_scale=lr_scale, 
                                                       initial_weights=initial_weights, **kwargs)
        else:
            self.weights = self.fit_predictor_lstsq(centers, targets, solver=solver)

    def fit_predictor_lstsq(self, centers, targets, solver='solve'):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        assert(len(centers)==len(targets))

        if centers.device != self.device:
            centers = centers.to(self.device)
            targets = targets.to(self.device)

        kernel_matrix = self.kernel(centers, centers)    

        if self.reg > 0:
            kernel_matrix.diagonal().add_(self.reg)
        
        
        if solver == 'solve':
            out = torch.linalg.solve(kernel_matrix, targets)
        elif solver == 'cholesky':
            L = torch.linalg.cholesky(kernel_matrix, out=kernel_matrix)
            out = torch.cholesky_solve(targets, L)
        elif solver == 'lu':
            P, L, U = torch.linalg.lu(kernel_matrix)
            out = torch.linalg.lu_solve(P, L, U, targets)
        else:
            raise ValueError(f"Invalid solver: {solver}")
        
        return out

    def fit_predictor_eigenpro(self, centers, targets, bs, lr_scale, initial_weights=None, **kwargs):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        n_classes = 1 if targets.dim()==1 else targets.shape[-1]
        ep_model = KernelModel(self.kernel, centers, n_classes, device=self.device)
        if initial_weights is not None:
            ep_model.weight = initial_weights.to(ep_model.weight.device, dtype=ep_model.weight.dtype)
        _ = ep_model.fit(centers, targets, verbose=self.verbose, mem_gb=self.mem_gb, bs=bs, 
                         lr_scale=lr_scale, classification=self.classification, **kwargs)
        return ep_model.weight.clone()

    @with_env_var("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    def predict(self, samples, max_batch_size=50_000):
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        samples, original_format = self.validate_samples(samples)
        out = []
        for i in range(0, samples.shape[0], max_batch_size):
            out_batch = self.kernel(samples[i:i+max_batch_size].to(self.device), self.centers.to(self.device)) @ self.weights.to(self.device)
            out.append(out_batch)
        out = torch.cat(out, dim=0)
        return self.convert_to_format(out, original_format)

    def validate_samples(self, samples):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        original_format = {}
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
            original_format['type'] = 'numpy'
            original_format['device'] = 'cpu'
        elif isinstance(samples, torch.Tensor):
            original_format['type'] = 'torch'
            original_format['device'] = samples.device
        else:
            raise ValueError(f"Invalid sample type: {type(samples)}")
        return samples.to(self.device), original_format
    
    def convert_to_format(self, tensor, original_format):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        if original_format['type'] == 'numpy':
            return tensor.cpu().numpy()
        elif original_format['type'] == 'torch':
            return tensor.to(original_format['device'])

    def validate_data(self, train_data, val_data):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        assert train_data is not None, "Train data must be provided"
        assert val_data is not None, "Validation data must be provided"

        X_train, y_train = train_data
        X_val, y_val = val_data

        X_train, _ = self.validate_samples(X_train)
        X_val, _ = self.validate_samples(X_val)
        y_train, _ = self.validate_samples(y_train)
        y_val, _ = self.validate_samples(y_val)

        if len(y_val.shape) == 1:
            y_val = y_val.unsqueeze(-1)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(-1)

        return X_train, y_train, X_val, y_val
    
    def adapt_params_to_data(self, n, d):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

        if self.tuning_metric == 'accuracy' and self.early_stop_rfm:
            if n <= 30_000:
                self.early_stop_multiplier = min(self.early_stop_multiplier, 1.003)
            else:
                self.early_stop_multiplier = min(self.early_stop_multiplier, 1.006)
            print(f"More aggressive early stop multiplier for accuracy: {self.early_stop_multiplier}")
            

        self.keep_device = d > n                                                        
        ep_epochs = 8
        total_points_to_sample = 20_000
        iters_to_use = 4
        if isinstance(self.kernel_obj, ProductLaplaceKernel):
            ep_epochs = 2
            if n > 1000:                                                               
                if n <= 10_000:
                                                               
                    pass
                elif 10_000 < n <= 20_000 and d <= 2000:
                                                                        
                    total_points_to_sample = min(total_points_to_sample, 10_000)
                    iters_to_use = min(iters_to_use, 4)
                elif 20_000 < n <= 50_000 and d <= 2000:
                                                                        
                    total_points_to_sample = min(total_points_to_sample, 2500)
                    iters_to_use = min(iters_to_use, 2)
                elif 10_000 < n <= 20_000 and d <= 3000:
                                                                      
                    total_points_to_sample = 2500
                    iters_to_use = min(iters_to_use, 2)
                elif d < 1000:
                                                                
                    total_points_to_sample = 2000
                    iters_to_use = min(iters_to_use, 1)
                elif d < 4000:
                                                                
                    total_points_to_sample = 1000
                    iters_to_use = min(iters_to_use, 1)
                else:
                                                
                    total_points_to_sample = 250
                    iters_to_use = min(iters_to_use, 1)
        if n >= 70_000:
                                                                           
            iters_to_use = min(iters_to_use, 2)

        ep_epochs = ep_epochs if self.ep_epochs is None else self.ep_epochs
        total_points_to_sample = total_points_to_sample if self.total_points_to_sample is None else self.total_points_to_sample
        iters_to_use = iters_to_use if self.iters is None else self.iters

        self.iters = iters_to_use
        self.total_points_to_sample = total_points_to_sample
        self.ep_epochs = ep_epochs
        return
    
    def _initialize_fit_parameters(self, iters, method, reg, verbose, M_batch_size, total_points_to_sample, 
                                   ep_epochs, tuning_metric, early_stop_rfm, early_stop_multiplier, 
                                   center_grads, prefit_eigenpro, **kwargs):
                                                       
        self.verbose = verbose if verbose is not None else self.verbose
        self.fit_using_eigenpro = (method.lower()=='eigenpro')
        self.prefit_eigenpro = prefit_eigenpro
        self.reg = reg if reg is not None else self.reg
        self.M_batch_size = M_batch_size
        self.total_points_to_sample = total_points_to_sample
        self.iters = iters if iters is not None else self.iters
        self.ep_epochs = ep_epochs
        self.tuning_metric = tuning_metric if tuning_metric is not None else self.tuning_metric
        self.minimize = not Metric.from_name(self.tuning_metric).should_maximize
        self.early_stop_rfm = early_stop_rfm
        self.early_stop_multiplier = early_stop_multiplier
        self.center_grads = center_grads
        self.top_k = kwargs.get('top_k', None)
        assert 'diag' not in kwargs, "diag should be set in the constructor"

    def _compute_validation_metrics(self, X_train, y_train, X_val, y_val, iteration_num=None, is_final=False, **kwargs):
                                                                

        metric = Metric.from_name(self.tuning_metric)
        if 'agop' in metric.required_quantities:
            self.agop = self.fit_M(X_train, y_train, inplace=False, **kwargs)
        val_metrics = self.score(X_val, y_val, metrics=[self.tuning_metric])
        if self.verbose:
            prefix = "Final" if is_final else f"Round {iteration_num}"
            print(f"{prefix} Val {metric.display_name}: {val_metrics[self.tuning_metric]:.4f}")
        return val_metrics

    def _should_early_stop(self, current_metric, best_metric):
                                                      
        if self.minimize:
            return current_metric > best_metric * self.early_stop_multiplier
        else:
            return current_metric < best_metric / self.early_stop_multiplier

                                                                                                               
    @with_env_var("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") 
    def fit(self, train_data, val_data=None, iters=None, method='lstsq', reg=None, center_grads=False,
            verbose=False, M_batch_size=None, ep_epochs=None, return_best_params=True, bs=None, 
            return_Ms=False, lr_scale=1, total_points_to_sample=None, solver='solve', 
            tuning_metric=None, prefit_eigenpro=True, early_stop_rfm=True, early_stop_multiplier=1.1, 
            callback=None, **kwargs):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

                               
        self._initialize_fit_parameters(iters, method, reg, verbose, M_batch_size, total_points_to_sample,
                                       ep_epochs, tuning_metric, early_stop_rfm, early_stop_multiplier,
                                       center_grads, prefit_eigenpro, **kwargs)
        
        

                                   
        X_train, y_train, X_val, y_val = self.validate_data(train_data, val_data)
        n, d = X_train.shape
        print("="*70)
        print(f"Fitting RFM with ntrain: {n}, d: {d}, and nval: {X_val.shape[0]}")
        print("="*70)


        if self.class_converter is None:
            self.class_converter = ClassificationConverter(mode='zero_one', n_classes=max(2, y_train.shape[1]))


        self.adapt_params_to_data(n, d)
        
                                       
        metrics, Ms = [], []
        best_alphas, best_M, best_sqrtM = None, None, None
        best_metric = float('inf') if self.tuning_metric == 'mse' else 0
        best_iter = None
        early_stopped = False
        best_bandwidth = self.kernel_obj.bandwidth+0

        start_time = time.time()

                            
        for i in range(self.iters):
                              
            if i > 0 and self.time_limit_s is not None and (i+1)/i*(time.time()-start_time) > self.time_limit_s:
                break                                                  


            if callback is not None:
                callback(iteration=i)

            start = time.time()
            self.fit_predictor(X_train, y_train, X_val=X_val, y_val=y_val, 
                               bs=bs, lr_scale=lr_scale, solver=solver, 
                               **kwargs)
                        
                                        
            val_metrics = self._compute_validation_metrics(X_train, y_train, X_val, y_val, iteration_num=i, **kwargs)

                                              
            if return_best_params:
                best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(
                    best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth, 
                    val_metrics[self.tuning_metric], i)
             
                                      
            if self.early_stop_rfm:
                val_metric = val_metrics[self.tuning_metric]
                if self._should_early_stop(val_metric, best_metric):
                    print(f"Early stopping at iteration {i}")
                    if not return_best_params:
                        self.fit_M(X_train, y_train.shape[-1], **kwargs)
                    early_stopped = True
                    break

                                      
            self.fit_M(X_train, y_train.shape[-1], **kwargs)
            del self.weights
            
            if return_Ms:
                Ms.append(self.tensor_copy(self.M))
                metrics.append(val_metrics[self.tuning_metric])

            print(f"Time taken for round {i}: {time.time() - start} seconds")

        if callback is not None:
            callback(iteration=self.iters)

                                                              
        if not early_stopped:
            self.fit_predictor(X_train, y_train, X_val=X_val, y_val=y_val, bs=bs, **kwargs)        
            final_val_metrics = self._compute_validation_metrics(X_train, y_train, X_val, y_val, is_final=True, **kwargs)

            if return_best_params:
                best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(
                    best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth, 
                    final_val_metrics[self.tuning_metric], iters)
                
                                 
        if return_best_params:
            self.M = None if best_M is None else best_M.to(self.device)
            self.sqrtM = None if best_sqrtM is None else best_sqrtM.to(self.device)
            self.weights = best_alphas.to(self.device)
            self.kernel_obj.bandwidth = best_bandwidth

        self.best_iter = best_iter

        if self.verbose:
            print(f"{self.best_iter=}")

        if kwargs.get('get_agop_best_model', False):
                                    
            self.agop_best_model = self.fit_M(X_train, y_train, inplace=False, **kwargs)

        return Ms if return_Ms else None
    
    def _compute_optimal_M_batch(self, n, c, d, scalar_size=4, mem_constant=2., max_batch_size=10_000, 
                            max_cheap_batch_size=20_000, light_kernels=Union[LaplaceKernel, LightLaplaceKernel]):
                                                       
        if self.device in ['cpu', torch.device('cpu')] or isinstance(self.kernel_obj, light_kernels):
                                                                                 
            M_batch_size = max(min(n, max_cheap_batch_size), 1)
        else:
            total_memory_possible = torch.cuda.get_device_properties(self.device).total_memory
            curr_mem_use = torch.cuda.memory_allocated()
            available_memory = total_memory_possible - curr_mem_use
            M_batch_size = int(available_memory / (mem_constant*n*c*d*scalar_size))
            M_batch_size = min(M_batch_size, max_batch_size)
        print(f"Optimal M batch size: {M_batch_size}")
        return M_batch_size
    
    def fit_M(self, samples, num_classes, M_batch_size=None, inplace=True, **kwargs):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        
        n, d = samples.shape
        M = torch.zeros_like(self.M) if self.M is not None else (
            torch.zeros(d, dtype=samples.dtype, device=self.device) 
            if self.diag else torch.zeros(d, d, dtype=samples.dtype, device=self.device))
        

        if M_batch_size is None: 
            BYTES_PER_SCALAR = samples.element_size()
            M_batch_size = self._compute_optimal_M_batch(n, num_classes, d, scalar_size=BYTES_PER_SCALAR)
        
        batches = torch.arange(n).split(M_batch_size)

        num_batches = 1 + self.total_points_to_sample//M_batch_size
        batches = batches[:num_batches]
        if self.verbose:
            print(f'Sampling AGOP on maximum of {num_batches*M_batch_size} total points')

        if self.verbose:
            for i, bids in tenumerate(batches):
                M.add_(self.update_M(samples[bids]))
        else:
            for bids in batches:
                M.add_(self.update_M(samples[bids]))
        
        scaled_M = M / (M.max() + 1e-30)
        if self.use_sqrtM:
            sqrtM = matrix_power(scaled_M, self.agop_power)
        else:
            sqrtM = None
        
        if inplace:
            self.M = scaled_M
            self.sqrtM = sqrtM
        else:
            return scaled_M
        
    def score(self, samples, targets, metrics):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

        metrics = Metrics(metrics)
        assert len(targets.shape) == 2 and targets.shape[1] >= 1
        kwargs = dict(top_k=self.top_k, y_true_reg=targets)
        if 'y_pred' in metrics.required_quantities:
            kwargs['y_pred'] = self.predict(samples.to(self.device))
        if 'y_pred_proba' in metrics.required_quantities:
            kwargs['y_pred_proba'] = self.predict_proba(samples.to(self.device))
        if 'agop' in metrics.required_quantities:
            kwargs['agop'] = self.agop
        if 'y_true_class' in metrics.required_quantities:
            kwargs['y_true_class'] = self.class_converter.numerical_to_labels(targets)

        return metrics.compute(**kwargs)
    
    @with_env_var("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    def predict_proba(self, samples, eps=1e-3):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        predictions = self.predict(samples) 
        return self.class_converter.numerical_to_probas(predictions, eps=eps)
