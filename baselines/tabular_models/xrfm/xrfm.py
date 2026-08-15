import sys
import time

import numpy as np
import torch

from xrfm.rfm_src import RFM, matrix_power
from tqdm import tqdm
import copy

from .rfm_src.class_conversion import ClassificationConverter
from .rfm_src.metrics import Metric
from .tree_utils import get_param_tree


class xRFM:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, rfm_params=None, min_subset_size=60_000,
                 max_depth=None, device=None, n_trees=1, n_tree_iters=0,
                 split_method='top_vector_agop_on_subset', tuning_metric=None,
                 categorical_info=None, default_rfm_params=None,
                 fixed_vector=None, callback=None, classification_mode='zero_one', time_limit_s=None, n_threads=None):
        self.min_subset_size = min_subset_size
        self.rfm_params = rfm_params
        self.max_depth = max_depth
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trees = None
        self.projections = None
        self.models = None
        self.n_trees = n_trees
        self.n_tree_iters = n_tree_iters
        self.tuning_metric = tuning_metric
        self.split_method = split_method
        self.maximizing_metric = False if tuning_metric is None else Metric.from_name(tuning_metric).should_maximize
        self.categorical_info = categorical_info
        self.fixed_vector = fixed_vector
        self.callback = callback
        self.classification_mode = classification_mode
        self.time_limit_s = time_limit_s
        self.n_threads = n_threads

                                                               
        self.min_val_size = 1500
        self.val_size_frac = 0.2

                                                          
        print(default_rfm_params)
        if default_rfm_params is None:
            self.default_rfm_params = {
                'model': {
                    "kernel": 'l2',
                    "exponent": 1.0,
                    "bandwidth": 10.0,
                    "diag": False,
                    "bandwidth_mode": "constant"
                },
                'fit': {
                    "get_agop_best_model": True,
                    "return_best_params": False,
                    "reg": 1e-3,
                    "iters": 0,
                    "early_stop_rfm": False,
                    "verbose": False
                }
            }
        else:
            self.default_rfm_params = default_rfm_params

        if self.rfm_params is None:
            self.rfm_params = self.default_rfm_params
            self.rfm_params['return_best_params'] = True

    def tree_copy(self, tree):
\
\
\
\
\
\
\
\
\
\
\
\
           
        return copy.deepcopy(tree)

    def _generate_random_projection(self, dim):
\
\
\
\
\
\
\
\
\
\
\
\
           
        projection = torch.randn(dim, device=self.device)
        return projection / torch.norm(projection)

    def _generate_projection_from_M(self, dim, M):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        if M.dim() == 1:                    
            std_devs = torch.sqrt(M)
            projection = torch.normal(0, std_devs).to(self.device)
        else:                         
                                                                      
            z = torch.randn(dim, device=self.device)

            try:
                sqrtM = matrix_power(M, 0.5)

                                                             
                projection = sqrtM @ z
            except:
                print(f"Matrix power failed, defaulting to random projection")

                                                                     
                projection = torch.randn(dim, device=self.device)

                                  
        return projection / torch.norm(projection)

    def _collect_leaf_nodes(self, node):
\
\
\
\
\
\
\
\
\
\
\
\
           
        if node['type'] == 'leaf':
            return [node]

        left_nodes = self._collect_leaf_nodes(node['left'])
        right_nodes = self._collect_leaf_nodes(node['right'])

        return left_nodes + right_nodes

    def _collect_attr(self, attr_name):
\
\
\
\
\
\
\
\
\
\
\
\
           
        best_agops = []
        for t in self.trees:
            leaf_nodes = self._collect_leaf_nodes(t)
            best_agops += [getattr(node['model'], attr_name) for node in leaf_nodes]
        return best_agops

    def collect_best_agops(self):
\
\
\
\
\
\
\
           
        return self._collect_attr('agop_best_model')
                         
                              
                                                      
                                                                                  
                           

    def collect_Ms(self):
\
\
\
\
\
\
\
           
        return self._collect_attr('M')

    def _average_M_across_leaves(self, tree):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        leaf_nodes = self._collect_leaf_nodes(tree)
        leaf_models = [node['model'] for node in leaf_nodes]

                                                 
        M_matrices = []
        for model in leaf_models:
            if hasattr(model, 'M') and model.M is not None:
                M_matrices.append(model.M)
            else:
                identity = torch.ones(self.data_dim, device=self.device) if model.diag else torch.eye(self.data_dim,
                                                                                                      device=self.device)
                M_matrices.append(identity)

        if M_matrices[0].dim() == 1:                    
            avg_M = torch.stack(M_matrices).mean(dim=0)
        else:                         
            avg_M = torch.stack(M_matrices).mean(dim=0)

        return avg_M

    def _get_balanced_split(self, projections, train_median):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
                       
        left_mask = projections < train_median
        right_mask = projections > train_median
        median_mask = projections == train_median

                                      
        n_left, n_right = left_mask.sum(), right_mask.sum()

                                                                       
        if n_left != n_right and median_mask.any():
            median_indices = torch.where(median_mask)[0]

            if n_left < n_right:
                                                 
                n_to_add = min(median_indices.size(0), n_right - n_left)
                left_mask[median_indices[:n_to_add]] = True
            else:
                                                  
                n_to_add = min(median_indices.size(0), n_left - n_right)
                right_mask[median_indices[:n_to_add]] = True

                                                                     
            if n_to_add > 0:
                median_mask[median_indices[:n_to_add]] = False

                                                                                     
        if median_mask.any():
            median_indices = torch.where(median_mask)[0]
            n_median = median_indices.size(0)
                                                                                          
            left_half = median_indices[:n_median // 2]
            right_half = median_indices[n_median // 2:]

                          
            left_mask[left_half] = True
            right_mask[right_half] = True

        assert not (left_mask & right_mask).any(), "Left and right masks should not overlap"
        assert left_mask.sum() - right_mask.sum() <= 1, "Left and right masks should have the same number of elements"

        return left_mask, right_mask

    def _build_tree(self, X, y, X_val, y_val, train_indices=None, depth=0, avg_M=None, is_root=False,
                    time_limit_s=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        start_time = time.time()
        n_samples = X.shape[0]
        if train_indices is None:
            train_indices = torch.arange(n_samples, device=self.device)

                                   
        if (n_samples <= self.min_subset_size) or (self.max_depth is not None and depth >= self.max_depth):
            if not is_root:                                                             
                print("Refilling validation set, because at least one split has been made.")
                X, y, X_val, y_val, train_indices = self._refill_val_set(X, y, X_val, y_val, train_indices)

                                                          
            model = RFM(**self.rfm_params['model'], tuning_metric=self.tuning_metric,
                        categorical_info=self.categorical_info, device=self.device, time_limit_s=time_limit_s,
                        **self.extra_rfm_params_)

            model.fit((X, y), (X_val, y_val), **self.rfm_params['fit'], callback=self.callback)
            return {'type': 'leaf', 'model': model, 'train_indices': train_indices, 'is_root': is_root}

                                    
        if avg_M is not None and self.split_method == 'random_global_agop':
            projection = self._generate_projection_from_M(X.shape[1], avg_M)
        elif self.split_method == 'random_pca':
            Xb = X - X.mean(dim=0, keepdim=True)
            Xcov = Xb.T @ Xb
            projection = self._generate_projection_from_M(X.shape[1], Xcov)
        elif self.split_method == 'linear':
            XtX = X.T @ X
            beta = torch.linalg.solve(XtX + 1e-6 * torch.eye(X.shape[1], device=self.device), X.T @ y)
            beta = beta.mean(dim=1)                                        
            projection = beta / torch.norm(beta)
        elif 'agop_on_subset' in self.split_method:
            print(f"Using {self.split_method} split method")
            sub_time_limit_s = None
            if time_limit_s is not None:
                                                                                                              
                n_leaves = 2 ** np.ceil(np.log2(n_samples / self.min_subset_size))
                sub_time_limit_s = 0.5 * time_limit_s / (n_leaves - 1)
            M = self._get_agop_on_subset(X, y, time_limit_s=sub_time_limit_s)
            if self.split_method == 'top_vector_agop_on_subset':
                                                
                _, _, Vt = torch.linalg.svd(M,
                                            full_matrices=False)                                                                 
                projection = Vt[0]
            elif self.split_method == 'random_agop_on_subset':
                projection = self._generate_projection_from_M(X.shape[1], M)
            elif self.split_method == 'top_pc_agop_on_subset':
                sqrtM = matrix_power(M, 0.5)
                XM = X @ sqrtM
                Xb = XM - XM.mean(dim=0, keepdim=True)
                                                            
                _, _, Vt = torch.linalg.svd(Xb.T @ Xb,
                                            full_matrices=False)                                               
                projection = Vt[0]
        elif self.split_method == 'fixed_vector':
            projection = self.fixed_vector
        else:
            projection = self._generate_random_projection(X.shape[1])

                                                
        projections = X @ projection

                                    
        train_median = torch.median(projections)

                                                                                            
        left_mask, right_mask = self._get_balanced_split(projections, train_median)

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

                                                      
        projections_val = X_val @ projection
        left_mask_val = projections_val <= train_median
        right_mask_val = ~left_mask_val

        X_val_left, y_val_left = X_val[left_mask_val], y_val[left_mask_val]
        X_val_right, y_val_right = X_val[right_mask_val], y_val[right_mask_val]

                        
        left_tree = self._build_tree(X_left, y_left, X_val_left, y_val_left,
                                     train_indices=train_indices[left_mask],
                                     depth=depth + 1,
                                     avg_M=avg_M,
                                     is_root=False,
                                     time_limit_s=None if time_limit_s is None
                                     else 0.5 * (time_limit_s - (time.time() - start_time)))
        right_tree = self._build_tree(X_right, y_right, X_val_right, y_val_right,
                                      train_indices=train_indices[right_mask],
                                      depth=depth + 1,
                                      avg_M=avg_M,
                                      is_root=False,
                                      time_limit_s=None if time_limit_s is None
                                      else time_limit_s - (time.time() - start_time)
                                      )

        return {
            'type': 'split',
            'split_direction': projection,
            'split_point': train_median,
            'left': left_tree,
            'right': right_tree,
            'is_root': is_root
        }

    def _refill_val_set(self, X, y, X_val, y_val, train_indices):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

        if len(X_val) <= self.min_val_size:
            n_orig_val = len(X_val)
            n_orig_train = len(X)

            num_val_to_add = self.min_val_size - len(X_val)
            num_val_to_add = min(num_val_to_add, int(len(X) * self.val_size_frac))
            shuffled_indices = torch.randperm(len(X))
            val_indices = shuffled_indices[:num_val_to_add]
            local_train_indices_to_keep = shuffled_indices[num_val_to_add:]

            X_val = torch.cat([X_val, X[val_indices]])
            y_val = torch.cat([y_val, y[val_indices]])
            X = X[local_train_indices_to_keep]
            y = y[local_train_indices_to_keep]

            train_indices = train_indices[local_train_indices_to_keep]

            assert n_orig_val + num_val_to_add == len(X_val) == len(y_val)
            assert n_orig_train - num_val_to_add == len(X) == len(y)

        return X, y, X_val, y_val, train_indices

    def _build_tree_with_iterations(self, X, y, X_val, y_val, time_limit_s=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        avg_M = None
        start_time = time.time()

                                                 
        tree = self._build_tree(X, y, X_val, y_val, avg_M=None, is_root=True,
                                time_limit_s=None if time_limit_s is None else time_limit_s / (1 + self.n_tree_iters))

                                                    
        best_val_score = self.score_tree(X_val, y_val, tree)
        best_tree = self.tree_copy(tree)

        val_scores = [best_val_score + 0]

        for iter in tqdm(range(self.n_tree_iters), desc="Iterating tree"):
            if time_limit_s is not None and (iter + 2) / (iter + 1) * (time.time() - start_time) > time_limit_s:
                break                                                         

                                                                       
            avg_M = self._average_M_across_leaves(tree)

            del tree

                                                      
            tree = self._build_tree(X, y, X_val, y_val, avg_M=avg_M, is_root=False,
                                    time_limit_s=None if time_limit_s is None
                                    else (time_limit_s - (time.time() - start_time)) / (self.n_tree_iters - iter))

                                                               
            val_score = self.score_tree(X_val, y_val, tree)
            val_scores.append(val_score)

            if self.maximizing_metric and val_score > best_val_score:
                best_val_score = val_score
                best_tree = self.tree_copy(tree)
            elif not self.maximizing_metric and val_score < best_val_score:
                best_val_score = val_score
                best_tree = self.tree_copy(tree)

        print("==========================Tree iteration results==========================")
        print("Validation scores over tree iterations:", val_scores)
        print("Best validation score:", best_val_score)
        print("==========================================================================")
        return best_tree

    def fit(self, X, y, X_val, y_val):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        print(f"Fitting xRFM with {self.n_trees} trees and {self.n_tree_iters} iterations per tree")

        if self.n_threads is not None:
            old_n_threads = torch.get_num_threads()
            torch.set_num_threads(self.n_threads)

                                            
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val, dtype=torch.float32, device=self.device)

        y = torch.as_tensor(y).to(self.device)
        y_val = torch.as_tensor(y_val).to(self.device)
        y_train_and_val = torch.cat([y, y_val], dim=0)

                                                                           
        if self.tuning_metric is not None:
            metric = Metric.from_name(self.tuning_metric)
            is_class = not ('reg' in metric.task_types)
            if is_class and y.is_floating_point():
                print(f'Warning: Using floating point y with a classification metric. '
                      f'Assuming that y is already binarized / one-hot encoded.', file=sys.stderr, flush=True)
        else:
            is_class = not y.is_floating_point()
            self.tuning_metric = 'brier' if is_class else 'mse'

                                                       
        if is_class:
            if y.is_floating_point():
                if len(y.shape) == 1:
                    y = y[:, None]
                assert len(y.shape) == 2

                self.n_classes_ = max(2, y.shape[1])
                self.class_converter_ = ClassificationConverter(mode=self.classification_mode,
                                                                n_classes=self.n_classes_)
            else:
                self.n_classes_ = max(2, y_train_and_val.max().item() + 1)

                self.class_converter_ = ClassificationConverter(mode=self.classification_mode, labels=y,
                                                                n_classes=self.n_classes_)

                y = self.class_converter_.labels_to_numerical(y)
                y_val = self.class_converter_.labels_to_numerical(y_val)

            self.extra_rfm_params_ = dict(class_converter=self.class_converter_)
        else:
            self.n_classes_ = 0
            y = y.float()
            y_val = y_val.float()

                                          
            if len(y.shape) == 1:
                y = y.unsqueeze(-1)
            if len(y_val.shape) == 1:
                y_val = y_val.unsqueeze(-1)
            assert len(y.shape) == 2
            self.extra_rfm_params_ = dict()

        self.data_dim = X.shape[1]

                              
        self.trees = []
        start_time = time.time()
        for iter in tqdm(range(self.n_trees), desc="Building trees"):
            if iter > 0 and self.time_limit_s is not None and (iter + 1) / iter * (
                    time.time() - start_time) > self.time_limit_s:
                break
            time_limit_s = None if self.time_limit_s is None else (self.time_limit_s - (time.time() - start_time)) / (
                    self.n_trees - iter)
            if self.n_tree_iters > 0:
                tree = self._build_tree_with_iterations(X, y, X_val, y_val,
                                                        time_limit_s=time_limit_s)
            else:
                tree = self._build_tree(X, y, X_val, y_val, is_root=True, time_limit_s=time_limit_s)
            self.trees.append(tree)

            if tree['type'] == 'leaf':
                print("Tree has no split, stopping training")
                break

        if self.n_threads is not None:
            torch.set_num_threads(old_n_threads)

        return self


    def score(self, samples, targets):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

        metric = Metric.from_name(self.tuning_metric)
        assert len(targets.shape) == 2 and targets.shape[1] >= 2
        kwargs = dict(y_true_reg=targets)
        if 'y_pred' in metric.required_quantities:
            kwargs['y_pred'] = self.predict(samples.to(self.device)).to(targets.device)
        if 'y_pred_proba' in metric.required_quantities:
            kwargs['y_pred_proba'] = self.predict_proba(samples.to(self.device)).to(targets.device)
        if 'y_true_class' in metric.required_quantities:
            kwargs['y_true_class'] = self.class_converter_.numerical_to_labels(targets)

        return metric.compute(**kwargs)


    def score_tree(self, samples, targets, tree):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

        metric = Metric.from_name(self.tuning_metric)
        assert len(targets.shape) == 2 and targets.shape[1] >= 2
        kwargs = dict(y_true_reg=targets)
        if 'y_pred' in metric.required_quantities:
            kwargs['y_pred'] = self._predict_tree(samples.to(self.device), tree).to(targets.device)
        if 'y_pred_proba' in metric.required_quantities:
            kwargs['y_pred_proba'] = self._predict_tree(samples.to(self.device), tree, proba=True).to(targets.device)
        if 'y_true_class' in metric.required_quantities:
            kwargs['y_true_class'] = self.class_converter_.numerical_to_labels(targets)

        return metric.compute(**kwargs)


    def predict(self, X):
\
\
\
\
\
\
\
\
\
\
\
\
           
        if self.trees is None:
            raise ValueError("Model has not been fitted yet.")

        if self.n_threads is not None:
            old_n_threads = torch.get_num_threads()
            torch.set_num_threads(self.n_threads)

                                           
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

        all_predictions = []

                                        
        for tree in self.trees:
            tree_predictions = self._predict_tree(X, tree)
            all_predictions.append(tree_predictions)

                                          
        pred = torch.mean(torch.stack(all_predictions), dim=0)

        if self.n_threads is not None:
            torch.set_num_threads(old_n_threads)

        if self.n_classes_ > 0:
            return self.class_converter_.numerical_to_labels(pred).cpu().numpy()
        else:
            return pred.cpu().numpy()


    def predict_proba(self, X):
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        if self.trees is None:
            raise ValueError("Model has not been fitted yet.")

        if self.n_threads is not None:
            old_n_threads = torch.get_num_threads()
            torch.set_num_threads(self.n_threads)

                                           
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        all_probas = []
        for tree in self.trees:
            tree_probas = self._predict_tree(X, tree, proba=True)
            all_probas.append(tree_probas)

        result = torch.mean(torch.stack(all_probas), dim=0)

        if self.n_threads is not None:
            torch.set_num_threads(old_n_threads)

        return result


    def _predict_tree(self, X, tree, proba=False):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

        X_leaf_groups, X_leaf_group_indices, leaf_nodes = self._get_leaf_groups_and_models_on_samples(X, tree)
        predictions = []
        for X_leaf, leaf_node in zip(X_leaf_groups, leaf_nodes):
            if proba:
                preds = leaf_node['model'].predict_proba(X_leaf)
            else:
                preds = leaf_node['model'].predict(X_leaf)
            predictions.append(preds)

        def reorder_tensor(original_tensor, order_tensor):
\
\
\
\
\
\
\
               
                                                        
                                                          
            _, sorted_indices = torch.sort(order_tensor)

                                                                   
            return original_tensor[sorted_indices]

        order = torch.cat(X_leaf_group_indices, dim=0)
        return reorder_tensor(torch.cat(predictions, dim=0), order)



    def load_state_dict(self, state_dict, X_train):
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        self.rfm_params = state_dict['rfm_params']
        self.categorical_info = state_dict['categorical_info']

        self._build_leaf_models_from_param_trees(state_dict['param_trees'])

                                     
        for tree in self.trees:
            assert tree['is_root']
            leaf_nodes = self._collect_leaf_nodes(tree)
            for leaf_node in leaf_nodes:
                leaf_model = leaf_node['model']
                leaf_center_indices = leaf_node['train_indices']
                leaf_model.centers = X_train[leaf_center_indices]
        return


    def _build_leaf_models_from_param_trees(self, param_trees):
\
\
\
\
\
\
\
\
\
\
\
           
        self.trees = []

        def set_leaf_model_single_tree(tree):
            if tree['type'] == 'leaf':
                leaf_model = RFM(**self.rfm_params['model'],
                                 categorical_info=self.categorical_info,
                                 device=self.device, **self.extra_rfm_params_)
                leaf_model.kernel_obj.bandwidth = tree['bandwidth']
                leaf_model.weights = tree['weights']
                leaf_model.M = tree['M']
                leaf_model.sqrtM = tree['sqrtM']
                tree['model'] = leaf_model
                return tree
            else:
                tree['left'] = set_leaf_model_single_tree(tree['left'])
                tree['right'] = set_leaf_model_single_tree(tree['right'])
                return tree

        for param_tree in param_trees:
            self.trees.append(set_leaf_model_single_tree(param_tree))

        return


    def get_state_dict(self):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        param_trees = []
        for tree in self.trees:
            param_trees.append(get_param_tree(tree, is_root=True))
        return {
            'rfm_params': self.rfm_params,
            'categorical_info': self.categorical_info,
            'param_trees': param_trees,
        }


    def _get_agop_on_subset(self, X, y, subset_size=50_000, time_limit_s=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        model = RFM(**self.default_rfm_params['model'], device=self.device, time_limit_s=time_limit_s,
                    **self.extra_rfm_params_)

        subset_size = min(subset_size, len(X))
        subset_train_size = int(subset_size * 0.95)                                                 

        subset_indices = torch.randperm(len(X))
        subset_train_indices = subset_indices[:subset_train_size]
        subset_val_indices = subset_indices[subset_train_size:subset_size]

        X_train = X[subset_train_indices]
        y_train = y[subset_train_indices]
        X_val = X[subset_val_indices]
        y_val = y[subset_val_indices]

        print("Getting AGOP on subset")
        print("X_train", X_train.shape, "y_train", y_train.shape, "X_val", X_val.shape, "y_val", y_val.shape)

        model.fit((X_train, y_train), (X_val, y_val), **self.default_rfm_params['fit'])
        agop = model.agop_best_model
        print("AGOP on subset", agop.shape)
        print("M", agop.diag()[:5])
        return agop


    def _get_leaf_groups_and_models_on_samples(self, X, tree):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
                                  
        X_leaf_groups = []
        X_leaf_group_indices = []
        leaf_nodes = []

                                                                    
        sample_indices = torch.arange(X.shape[0], device=self.device)
        stack = [(X, sample_indices, tree)]

                                         
        while stack:
            current_X, current_indices, current_node = stack.pop()

                                                             
            if current_node['type'] == 'leaf':
                X_leaf_groups.append(current_X)
                X_leaf_group_indices.append(current_indices)
                leaf_nodes.append(current_node)
                continue

                                                              
            projections = current_X @ current_node['split_direction']

                                                      
            left_mask = projections <= current_node['split_point']
            right_mask = ~left_mask

                                                                                        
            if right_mask.sum() > 0:
                stack.append((
                    current_X[right_mask],
                    current_indices[right_mask],
                    current_node['right']
                ))

                                     
            if left_mask.sum() > 0:
                stack.append((
                    current_X[left_mask],
                    current_indices[left_mask],
                    current_node['left']
                ))

        return X_leaf_groups, X_leaf_group_indices, leaf_nodes
