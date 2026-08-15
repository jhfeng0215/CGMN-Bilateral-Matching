from __future__ import annotations

import sys
import random
import itertools
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
)
from sklearn.utils.validation import check_is_fitted


class RecursionLimitManager:
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, limit):
        self.limit = limit
        self.original_limit = None

    def __enter__(self):
        self.original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)
        return self

    def __exit__(self, type, value, traceback):
        sys.setrecursionlimit(self.original_limit)
        return False                                        


class TransformToNumerical(TransformerMixin, BaseEstimator):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def fit(self, X, y=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        if not hasattr(X, "columns"):                                                                        
                          
            self.tfm_ = FunctionTransformer()
            return self

        cat_cols = make_column_selector(dtype_include=["string", "object", "category", "boolean"])(X)
        cat_pos = [X.columns.get_loc(col) for col in cat_cols]

        numeric_cols = make_column_selector(dtype_include="number")(X)
        numeric_pos = [X.columns.get_loc(col) for col in numeric_cols]

        self.tfm_ = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OrdinalEncoder(
                        dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
                    ),
                    cat_pos,
                ),
                ("continuous", SimpleImputer(), numeric_pos),
            ]
        )
        self.tfm_.fit(X)

        if self.verbose:
            selected_cols = []
            for name, tfm, pos in self.tfm_.transformers_:
                if tfm != "drop":
                    cols = list(X.columns[pos])
                    selected_cols.extend(cols)
                    print(f"Columns classified as {name}: {cols}")

            dropped_cols = set(X.columns).difference(set(selected_cols))
            if len(dropped_cols) >= 1:
                print(f"The following columns are not used due to their data type: {list(dropped_cols)}")

        return self

    def transform(self, X):
\
\
\
\
\
\
\
\
\
\
\
           
        return self.tfm_.transform(X)


class UniqueFeatureFilter(TransformerMixin, BaseEstimator):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, threshold: int = 1):
        self.threshold = threshold

    def fit(self, X, y=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        X = self._validate_data(X)

                                                          
        if X.shape[0] <= self.threshold:
            self.features_to_keep_ = np.ones(self.n_features_in_, dtype=bool)
        else:
                                                                                 
            self.features_to_keep_ = np.array(
                [len(np.unique(X[:, i])) > self.threshold for i in range(self.n_features_in_)]
            )

        self.n_features_out_ = np.sum(self.features_to_keep_)

        return self

    def transform(self, X):
\
\
\
\
\
\
\
\
\
\
\
           
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        return X[:, self.features_to_keep_]


class OutlierRemover(TransformerMixin, BaseEstimator):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold

    def fit(self, X, y=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        X = self._validate_data(X)

                                                                 
        self.means_ = np.nanmean(X, axis=0)
        self.stds_ = np.nanstd(X, axis=0, ddof=1 if X.shape[0] > 1 else 0)

                                                 
        self.stds_ = np.maximum(self.stds_, 1e-6)

                                                           
        X_clean = X.copy()
        lower_bounds = self.means_ - self.threshold * self.stds_
        upper_bounds = self.means_ + self.threshold * self.stds_

                                                
        lower_mask = X < lower_bounds[np.newaxis, :]
        upper_mask = X > upper_bounds[np.newaxis, :]
        outlier_mask = np.logical_or(lower_mask, upper_mask)

                             
        X_clean[outlier_mask] = np.nan

                                                             
        self.means_ = np.nanmean(X_clean, axis=0)
        self.stds_ = np.nanstd(X_clean, axis=0, ddof=1 if X.shape[0] > 1 else 0)

                                                 
        self.stds_ = np.maximum(self.stds_, 1e-6)

                              
        self.lower_bounds_ = self.means_ - self.threshold * self.stds_
        self.upper_bounds_ = self.means_ + self.threshold * self.stds_

        return self

    def transform(self, X):
\
\
\
\
\
\
\
\
\
\
\
           
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X)
        X = np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X)

        return X


class CustomStandardScaler(TransformerMixin, BaseEstimator):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, clip_min: float = -100, clip_max: float = 100, epsilon: float = 1e-6):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon

    def fit(self, X, y=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        X = self._validate_data(X)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + self.epsilon

        return self

    def transform(self, X):
\
\
\
\
\
\
\
\
\
\
\
           
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        X_scaled = (X - self.mean_) / self.scale_
        X_clipped = np.clip(X_scaled, self.clip_min, self.clip_max)

        return X_clipped


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
        noise: float = 1e-3,
        n_quantiles: int = 1000,
        subsample: int = 1_000_000_000,
        output_distribution: str = "normal",
        random_state: Optional[int] = None,
    ):
        self.noise = noise
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self.random_state = random_state

    def fit(self, X, y=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
                                                              
        n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

                                        
        normalizer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=n_quantiles,
            subsample=self.subsample,
            random_state=self.random_state,
        )

                               
        X_modified = self._add_noise(X) if self.noise > 0 else X

                            
        normalizer.fit(X_modified)

                               
        self.normalizer_ = normalizer

        return self

    def transform(self, X, y=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        check_is_fitted(self)
        return self.normalizer_.transform(X)

    def _add_noise(self, X):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        stds = np.std(X, axis=0, keepdims=True)
        noise_std = self.noise / np.maximum(stds, self.noise)
        rng = np.random.default_rng(self.random_state)
        X_noisy = X + noise_std * rng.standard_normal(X.shape)
        return X_noisy


class PreprocessingPipeline(TransformerMixin, BaseEstimator):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
        self, normalization_method: str = "power", outlier_threshold: float = 4.0, random_state: Optional[int] = None
    ):
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        X = self._validate_data(X)

                                   
        self.standard_scaler_ = CustomStandardScaler()
        X_scaled = self.standard_scaler_.fit_transform(X)

                                
        if self.normalization_method != "none":
            if self.normalization_method == "power":
                self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True)
            elif self.normalization_method == "quantile":
                self.normalizer_ = QuantileTransformer(output_distribution="normal", random_state=self.random_state)
            elif self.normalization_method == "quantile_rtdl":
                self.normalizer_ = Pipeline(
                    [
                        (
                            "quantile_rtdl",
                            RTDLQuantileTransformer(output_distribution="normal", random_state=self.random_state),
                        ),
                        ("std", StandardScaler()),
                    ]
                )
            elif self.normalization_method == "robust":
                self.normalizer_ = RobustScaler(unit_variance=True)
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")

            self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
            self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
            X_normalized = self.normalizer_.fit_transform(X_scaled)
        else:
            self.normalizer_ = None
            X_normalized = X_scaled

                            
        self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
        self.X_transformed_ = self.outlier_remover_.fit_transform(X_normalized)

        return self

    def transform(self, X):
\
\
\
\
\
\
\
\
\
\
\
           
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, copy=True)
                          
        X = self.standard_scaler_.transform(X)
                       
        if self.normalizer_ is not None:
            try:
                                                                                                       
                X = self.normalizer_.transform(X)
            except ValueError:
                                              
                X = np.clip(X, self.X_min_, self.X_max_)
                X = self.normalizer_.transform(X)
                         
        X = self.outlier_remover_.transform(X)

        return X


class FeatureShuffler:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
        n_features: int,
        method: str = "latin",
        max_features_for_latin: int = 4000,
        random_state: Optional[int] = None,
    ):
        self.n_features = n_features
        self.method = method
        self.max_features_for_latin = max_features_for_latin
        self.random_state = random_state

    def shuffle(self, n_estimators: int) -> List[np.ndarray]:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

        self.rng_ = random.Random(self.random_state)
        feature_indices = list(range(self.n_features))

                                                                                
        if self.n_features > self.max_features_for_latin and self.method == "latin":
            method = "random"
        else:
            method = self.method

                      
        if method == "none" or n_estimators == 1:
            shuffle_patterns = [feature_indices]

                                               
        if method == "shift":
                                          
            shuffle_patterns = [feature_indices[-i:] + feature_indices[:-i] for i in range(self.n_features)]
        elif method == "random":
                                 
            if self.n_features <= 5:
                all_perms = [list(perm) for perm in itertools.permutations(feature_indices)]
                shuffle_patterns = self.rng_.sample(all_perms, min(n_estimators, len(all_perms)))
            else:
                shuffle_patterns = [self.rng_.sample(feature_indices, self.n_features) for _ in range(n_estimators)]
        elif method == "latin":
                                       
            with RecursionLimitManager(100000):                                                         
                shuffle_patterns = self._latin_squares()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shift', 'random', 'latin', or 'none'.")

        return shuffle_patterns

    def _latin_squares(self):
\
\
\
\
\
\
           

        def _shuffle_transpose_shuffle(matrix):
            square = deepcopy(matrix)
            self.rng_.shuffle(square)
            trans = list(zip(*square))
            self.rng_.shuffle(trans)
            return trans

        def _rls(symbols):
            n = len(symbols)
            if n == 1:
                return [symbols]
            else:
                sym = self.rng_.choice(symbols)
                symbols.remove(sym)
                square = _rls(symbols)
                square.append(square[0].copy())
                for i in range(n):
                    square[i].insert(i, sym)
                return square

        symbols = list(range(self.n_features))
        square = _rls(symbols)
        feature_shuffles = _shuffle_transpose_shuffle(square)

        return [list(shuffle) for shuffle in feature_shuffles]


class EnsembleGenerator(TransformerMixin, BaseEstimator):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
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
        n_estimators: int,
        norm_methods: str | List[str] | None = None,
        feat_shuffle_method: str = "latin",
        class_shift: bool = True,
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shift = class_shift
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        self._validate_data(X, y)

        if self.norm_methods is None:
            self.norm_methods_ = ["none", "power"]
        else:
            if isinstance(self.norm_methods, str):
                self.norm_methods_ = [self.norm_methods]
            else:
                self.norm_methods_ = self.norm_methods

                                
        self.unique_filter_ = UniqueFeatureFilter()
        X = self.unique_filter_.fit_transform(X)

        self.X_ = X
        self.y_ = y

                                                                         
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))

        self.rng_ = random.Random(self.random_state)
        self.ensemble_configs_, self.feature_shuffle_patterns_, self.class_shift_offsets_ = self._generate_ensemble()

        self.preprocessors_ = {}
        for norm_method in self.ensemble_configs_:
            if norm_method not in self.preprocessors_:
                preprocessor = PreprocessingPipeline(
                    normalization_method=norm_method,
                    outlier_threshold=self.outlier_threshold,
                    random_state=self.random_state,
                )
                preprocessor.fit(X)
                self.preprocessors_[norm_method] = preprocessor

        return self

    def _generate_ensemble(self):
\
\
\
\
\
\
\
\
\
           

        shuffler = FeatureShuffler(
            n_features=self.n_features_in_, method=self.feat_shuffle_method, random_state=self.random_state
        )
        shuffle_patterns = shuffler.shuffle(self.n_estimators)

        if self.class_shift and self.n_estimators > 1:
            shift_offsets = self.rng_.sample(range(self.n_classes_), self.n_classes_)
        else:
            shift_offsets = [0]

        shuffle_shift_configs = list(itertools.product(shuffle_patterns, shift_offsets))
        self.rng_.shuffle(shuffle_shift_configs)

        shuffle_shift_norm_configs = list(itertools.product(shuffle_shift_configs, self.norm_methods_))
        shuffle_shift_norm_configs = shuffle_shift_norm_configs[: self.n_estimators]

                                                                                                  
        used_methods = list(set([config[1] for config in shuffle_shift_norm_configs]))

        ensemble_configs = OrderedDict()
        shuffle_patterns = OrderedDict()
        shift_offsets = OrderedDict()

        for method in used_methods:
            shuffle_shift_configs = [config[0] for config in shuffle_shift_norm_configs if config[1] == method]
            shuffle_patterns[method] = [config[0] for config in shuffle_shift_configs]
            shift_offsets[method] = [config[1] for config in shuffle_shift_configs]
            ensemble_configs[method] = shuffle_shift_configs

        return ensemble_configs, shuffle_patterns, shift_offsets

    def transform(self, X):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           

        check_is_fitted(self, ["ensemble_configs_"])

                                  
        X = self.unique_filter_.transform(X)
        y = self.y_

        data = OrderedDict()
        for norm_method, shuffle_shift_configs in self.ensemble_configs_.items():
                                 
            preprocessor = self.preprocessors_[norm_method]
            X_variant = np.concatenate(
                [preprocessor.X_transformed_, preprocessor.transform(X)],
                axis=0,
            )
                                                     
            X_ensemble = []
            y_ensemble = []
            for shuffle_pattern, shift_offset in shuffle_shift_configs:
                X_ensemble.append(X_variant[:, shuffle_pattern])
                y_ensemble.append((y + shift_offset) % self.n_classes_)
            data[norm_method] = (np.stack(X_ensemble, axis=0), np.stack(y_ensemble, axis=0))

        return data