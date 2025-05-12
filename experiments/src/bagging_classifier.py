"""Implementation of BaggingClassifier with oversampling and class ratio."""

import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .utils import RANDOM_SEED

warnings.simplefilter("ignore")
np.random.seed(RANDOM_SEED)


# pylint: disable=all
class BaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom BaggingClassifier to handle the case when the base estimator is not fitted.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        bootstrap: bool = True,
        random_state: int = None,
        oversampler_class: Any = None,
        threads: int = 4,
    ):
        """
        Initialize the BaggingClassifier.

        Args:
            base_estimator (BaseEstimator): The base estimator to fit on random subsets of the data.
            n_estimators (int): The number of base estimators in the ensemble.
            max_samples (float): The fraction of samples to draw from X to train each base estimator.
            bootstrap (bool): Whether samples are drawn with replacement.
            random_state (int): Random seed for reproducibility.
            oversampler_class (Any): Class for oversampling.
            threads (int): Number of threads to use for parallel processing.
        Raises:
            ValueError: If n_estimators <= 0, max_samples not in (0, 1], or threads <= 0.
        """
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0.")
        if not (0 < max_samples <= 1):
            raise ValueError("max_samples must be in (0, 1].")
        if threads <= 0:
            raise ValueError("threads must be greater than 0.")

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.oversampler_class = oversampler_class
        self.threads = threads

        self.classes_ = None  # support for sklearn

        self._fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        """
        Fit the BaggingClassifier to the training data.
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target labels.
        Returns:
            BaggingClassifier: Fitted BaggingClassifier instance.
        """
        if self._fitted_:
            raise ValueError("This BaggingClassifier instance is already fitted.")

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self.random_state_ = np.random.RandomState(self.random_state)

        # logic for oversampling with class ratio
        if len(self.classes_) != 2:
            raise ValueError(
                "This bagging classifier only supports binary classification."
            )
        ratios = self.get_class_ratio(y)

        with ThreadPoolExecutor(self.threads) as executor:
            futures = [
                executor.submit(
                    self._train_model,
                    X=X,
                    y=y,
                    ratios=ratios,
                    idx=i,
                )
                for i in range(self.n_estimators)
            ]
            self.estimators_ = [f.result() for f in futures]

        self._fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the input samples.
        Args:
            X (np.ndarray): Input samples.
        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.vote(X, proba_=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class probabilities for the input samples.
        Args:
            X (np.ndarray): Input samples.
        Returns:
            np.ndarray: Predicted class probabilities.
        """
        return self.vote(X, proba_=True)

    def ratio_aware_bootstrap(
        self, X: np.ndarray, y: np.ndarray, ratios: List[Tuple[np.ndarray, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a bootstrap sample with class ratio awareness.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Input labels.
            ratios (List[Tuple[np.ndarray, int]]): Class indices and their respective sample sizes.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Bootstrap sample of features and labels.
        """
        indices = np.concatenate(
            [
                self.random_state_.choice(idx, size=n, replace=self.bootstrap)
                for idx, n in ratios
            ]
        )
        return X[indices], y[indices]

    def get_class_ratio(self, y: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        Calculate the class ratio for oversampling.

        Args:
            y (np.ndarray): Input labels.
        Returns:
            List[Tuple[np.ndarray, int]]: Class indices and
        """
        ratios = []
        for k in self.classes_:
            cur_class_idx = np.where(y == k)[0]
            ratios.append((cur_class_idx, int(len(cur_class_idx) * self.max_samples)))
        return ratios

    def _predict(self, X: np.ndarray, proba_: bool = False) -> np.ndarray:
        """
        Predict the class labels for the input samples using all estimators.

        Args:
            X (np.ndarray): Input samples.
            proba_ (bool): If True, return class probabilities.
        Returns:
            np.ndarray: Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)

        with ThreadPoolExecutor(self.threads) as executor:
            futures = [executor.submit(est.predict, X) for est in self.estimators_]
            preds = np.array([f.result() for f in futures])

        return self.vote(preds, proba_)

    def vote(self, preds: np.ndarray, proba_: bool = False) -> np.ndarray:
        """
        Perform majority voting or soft voting for preds.

        Args:
            X (np.ndarray): Input samples.
            proba_ (bool): If True, return class probabilities.
        Returns:
            np.ndarray: Predicted class labels or probabilities.
        """
        if not proba_:
            # Use majority voting
            return np.apply_along_axis(
                lambda x: np.bincount(x, minlength=len(self.classes_)).argmax(),
                axis=0,
                arr=preds,
            )
        return np.apply_along_axis(
            lambda x: np.bincount(x, minlength=len(self.classes_)) / len(x),
            axis=0,
            arr=preds,
        ).T

    def _train_model(
        self,
        idx: int,
        X: np.ndarray,
        y: np.ndarray,
        ratios: List[Tuple[np.ndarray, int]],
        X_synth: np.ndarray | None = None,
        y_synth: np.ndarray | None = None,
    ) -> BaseEstimator:
        """
        Trains a single instance of the base classifier using training data from a cross-validation fold
        and optional synthetic data.

        Optional oversampling is exclusive with synthetic data.

        Args:
            i (int): Index of the estimator.
            X (np.ndarray): Training data.
            y (np.ndarray): Target labels.
            ratios (List[Tuple[np.ndarray, int]]): Class ratios for the training data.
            X_synth (np.ndarray | None): Optional synthetic data to include in training.
            y_synth (np.ndarray | None): Optional labels for the synthetic data.
        Returns:
            BaseEstimator: A trained instance (deep-copied) of the base classifier.
        """
        X_sample, y_sample = self.ratio_aware_bootstrap(X=X, y=y, ratios=ratios)

        if X_synth is not None:
            X_sample = np.vstack([X_sample, X_synth])
            y_sample = np.hstack([y_sample, y_synth])
        elif self.oversampler_class is not None:
            oversampler = self.oversampler_class(
                random_state=self.random_state + 10000 + idx
            )
            X_sample, y_sample = oversampler.fit_resample(X_sample, y_sample)

        estimator = clone(self.base_estimator)
        estimator.fit(X_sample, y_sample)
        return estimator

    def __sklearn_is_fitted__(self) -> bool:
        """
        Check if the estimator is fitted.
        Returns:
            bool: True if fitted, False otherwise.
        """
        return self._fitted_
