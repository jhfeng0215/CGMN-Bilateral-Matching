from __future__ import annotations

import warnings
from pathlib import Path
from packaging import version
from typing import Optional, List, Dict

import numpy as np
import torch

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder

from .preprocessing import TransformToNumerical, EnsembleGenerator
from .model.tabicl import InferenceConfig
from .model.tabicl import TabICL


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
OLD_SKLEARN = version.parse(sklearn.__version__) < version.parse("1.6")


class TabICLClassifier(ClassifierMixin, BaseEstimator):


    def __init__(
        self,
        n_estimators: int = 32,
        norm_methods: Optional[str | List[str]] = None,
        feat_shuffle_method: str = "latin",
        class_shift: bool = True,
        outlier_threshold: float = 4.0,
        softmax_temperature: float = 0.9,
        average_logits: bool = True,
        use_hierarchical: bool = True,
        use_amp: bool = True,
        batch_size: Optional[int] = 8,
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "tabicl-classifier-v1.1-0506.ckpt",
        device: Optional[str | torch.device] = None,
        random_state: int | None = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[InferenceConfig | Dict] = None,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shift = class_shift
        self.outlier_threshold = outlier_threshold
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.use_hierarchical = use_hierarchical
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.inference_config = inference_config

    def _more_tags(self):

        return dict(non_deterministic=True)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

    def _load_model(self):


        repo_id = "jingang/TabICL-clf"
        filename = self.checkpoint_version

        ckpt_legacy = "tabicl-classifier.ckpt"
        ckpt_v1 = "tabicl-classifier-v1-0208.ckpt"
        ckpt_v1_1 = "tabicl-classifier-v1.1-0506.ckpt"

        if filename == ckpt_legacy:
            info_message = (
                f"INFO: You are using '{ckpt_legacy}'. This is a legacy alias for '{ckpt_v1}' "
                f"and is maintained for backward compatibility. It may be removed in a future release.\n"
                f"Please consider using '{ckpt_v1}' or the latest '{ckpt_v1_1}' directly.\n"
                f"'{ckpt_legacy}' (effectively '{ckpt_v1}') is the version "
                f"used in the original TabICL paper. For improved performance, consider using '{ckpt_v1_1}'.\n"
            )
        elif filename == ckpt_v1:
            info_message = (
                f"INFO: You are downloading '{ckpt_v1}', the version used in the original TabICL paper.\n"
                f"A newer version, '{ckpt_v1_1}', is available and offers improved performance.\n"
            )
        elif filename == ckpt_v1_1:
            info_message = (
                f"INFO: You are downloading '{ckpt_v1_1}', the latest best-performing version of TabICL.\n"
                f"To reproduce results from the original paper, please use '{ckpt_v1}'.\n"
            )
        else:
            raise ValueError(
                f"Invalid checkpoint version '{filename}'. Available ones are: '{ckpt_legacy}', '{ckpt_v1}', '{ckpt_v1_1}'."
            )

        if self.model_path is None:

            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import LocalEntryNotFoundError
            try:
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
            except LocalEntryNotFoundError:
                if self.allow_auto_download:
                    print(info_message)
                    print(f"Checkpoint '{filename}' not cached.\n Downloading from Hugging Face Hub ({repo_id}).\n")
                    model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
                else:
                    raise ValueError(
                        f"Checkpoint '{filename}' not cached and automatic download is disabled.\n"
                        f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({repo_id})."
                    )
            if model_path_:
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
        else:

            model_path_ = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
            if model_path_.exists():

                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
            else:

                from huggingface_hub import hf_hub_download
                if self.allow_auto_download:
                    print(info_message)
                    print(
                        f"Checkpoint not found at '{model_path_}'.\n"
                        f"Downloading '{filename}' from Hugging Face Hub ({repo_id}) to this location.\n"
                    )
                    model_path_.parent.mkdir(parents=True, exist_ok=True)
                    cache_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path_.parent)
                    Path(cache_path).rename(model_path_)
                    checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                else:
                    raise ValueError(
                        f"Checkpoint not found at '{model_path_}' and automatic download is disabled.\n"
                        f"Either provide a valid checkpoint path, or set allow_auto_download=True to download "
                        f"'{filename}' from Hugging Face Hub ({repo_id})."
                    )

        assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
        assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."

        self.model_path_ = model_path_
        self.model_ = TabICL(**checkpoint["config"])
        self.model_.load_state_dict(checkpoint["state_dict"])
        self.model_.eval()

    def fit(self, X, y):


        if OLD_SKLEARN:

            X, y = self._validate_data(X, y, dtype=None, cast_to_ndarray=False)
        else:
            X, y = self._validate_data(X, y, dtype=None, skip_check_array=True)

        check_classification_targets(y)

        if self.device is None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device_ = torch.device(self.device)
        else:
            self.device_ = self.device


        self._load_model()
        self.model_.to(self.device_)


        init_config = {
            "COL_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
            "ROW_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
            "ICL_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
        }

        if self.inference_config is None:
            self.inference_config_ = InferenceConfig()
            self.inference_config_.update_from_dict(init_config)

        elif isinstance(self.inference_config, dict):
            self.inference_config_ = InferenceConfig()
            for key, value in self.inference_config.items():
                if key in init_config:
                    init_config[key].update(value)
            self.inference_config_.update_from_dict(init_config)

        else:
            self.inference_config_ = self.inference_config


        self.y_encoder_ = LabelEncoder()
        y = self.y_encoder_.fit_transform(y)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.y_encoder_.classes_)

        if self.n_classes_ > self.model_.max_classes and not self.use_hierarchical:
            raise ValueError(
                f"The number of classes ({self.n_classes_}) exceeds the max number of classes ({self.model_.max_classes}) "
                f"natively supported by the model. Consider enabling hierarchical classification."
            )

        if self.n_classes_ > self.model_.max_classes and self.verbose:
            print(
                f"The number of classes ({self.n_classes_}) exceeds the max number of classes ({self.model_.max_classes}) "
                f"natively supported by the model. Therefore, hierarchical classification is used."
            )


        self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
        X = self.X_encoder_.fit_transform(X)


        self.ensemble_generator_ = EnsembleGenerator(
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods or ["none", "power"],
            feat_shuffle_method=self.feat_shuffle_method,
            class_shift=self.class_shift,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X, y)

        return self

    def _batch_forward(self, Xs, ys, shuffle_patterns=None):


        batch_size = self.batch_size or Xs.shape[0]
        n_batches = np.ceil(Xs.shape[0] / batch_size)
        Xs = np.array_split(Xs, n_batches)
        ys = np.array_split(ys, n_batches)
        if shuffle_patterns is None:
            shuffle_patterns = [None] * n_batches
        else:
            shuffle_patterns = np.array_split(shuffle_patterns, n_batches)

        outputs = []
        for X_batch, y_batch, pattern_batch in zip(Xs, ys, shuffle_patterns):
            X_batch = torch.from_numpy(X_batch).float().to(self.device_)
            y_batch = torch.from_numpy(y_batch).float().to(self.device_)
            if pattern_batch is not None:
                pattern_batch = pattern_batch.tolist()

            with torch.no_grad():
                out = self.model_(
                    X_batch,
                    y_batch,
                    feature_shuffles=pattern_batch,
                    return_logits=True if self.average_logits else False,
                    softmax_temperature=self.softmax_temperature,
                    inference_config=self.inference_config_,
                )
            outputs.append(out.float().cpu().numpy())

        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X):

        check_is_fitted(self)
        if isinstance(X, np.ndarray) and len(X.shape) == 1:

            raise ValueError(f"The provided input X is one-dimensional. Reshape your data.")

        if self.n_jobs is not None:
            assert self.n_jobs != 0
            old_n_threads = torch.get_num_threads()

            import multiprocessing as mp

            n_logical_cores = mp.cpu_count()

            if self.n_jobs > 0:
                if self.n_jobs > n_logical_cores:
                    warnings.warn(
                        f"TabICL got n_jobs={self.n_jobs} but there are only {n_logical_cores} logical cores available."
                        f" Only {n_logical_cores} threads will be used."
                    )
                n_threads = max(n_logical_cores, self.n_jobs)
            else:
                n_threads = max(1, mp.cpu_count() + 1 + self.n_jobs)

            torch.set_num_threads(n_threads)


        if OLD_SKLEARN:

            X = self._validate_data(X, reset=False, dtype=None, cast_to_ndarray=False)
        else:
            X = self._validate_data(X, reset=False, dtype=None, skip_check_array=True)

        X = self.X_encoder_.transform(X)

        data = self.ensemble_generator_.transform(X)
        outputs = []
        for norm_method, (Xs, ys) in data.items():
            shuffle_patterns = self.ensemble_generator_.feature_shuffle_patterns_[norm_method]
            outputs.append(self._batch_forward(Xs, ys, shuffle_patterns))
        outputs = np.concatenate(outputs, axis=0)


        class_shift_offsets = []
        for offsets in self.ensemble_generator_.class_shift_offsets_.values():
            class_shift_offsets.extend(offsets)



        n_estimators = len(class_shift_offsets)


        avg = None
        for i, offset in enumerate(class_shift_offsets):
            out = outputs[i]

            out = np.concatenate([out[..., offset:], out[..., :offset]], axis=-1)

            if avg is None:
                avg = out
            else:
                avg += out


        avg /= n_estimators


        if self.average_logits:
            avg = self.softmax(avg, axis=-1, temperature=self.softmax_temperature)

        if self.n_jobs is not None:
            torch.set_num_threads(old_n_threads)


        return avg / avg.sum(axis=1, keepdims=True)

    def predict(self, X):

        proba = self.predict_proba(X)
        y = np.argmax(proba, axis=1)

        return self.y_encoder_.inverse_transform(y)

    @staticmethod
    def softmax(x, axis: int = -1, temperature: float = 0.9):

        x = x / temperature

        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)

        return e_x / np.sum(e_x, axis=axis, keepdims=True)