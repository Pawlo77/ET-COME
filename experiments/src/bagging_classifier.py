"""Implementation of BaggingClassifier with oversampling and class ratio."""

import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        """
        Fit the BaggingClassifier to the training data.
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target labels.
        Returns:
            BaggingClassifier: Fitted BaggingClassifier instance.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self.random_state_ = np.random.RandomState(self.random_state)

        # logic for oversampling with class ratio
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                "This bagging classifier only supports binary classification."
            )
        idx_0 = np.where(y == classes[0])[0]
        idx_1 = np.where(y == classes[1])[0]
        n0 = int(len(idx_0) * self.max_samples)
        n1 = int(len(idx_1) * self.max_samples)

        def fit_estimator(i: int) -> BaseEstimator:
            """
            Fit a single base estimator on a random subset of the data.
            Args:
                i (int): Index of the estimator.
            Returns:
                BaseEstimator: Fitted base estimator.
            """
            # Bootstrap sample
            # Maintain the original class ratio in the sampled indices.
            indices_0 = self.random_state_.choice(
                idx_0, size=n0, replace=self.bootstrap
            )
            indices_1 = self.random_state_.choice(
                idx_1, size=n1, replace=self.bootstrap
            )
            indices = np.concatenate([indices_0, indices_1])
            X_sample, y_sample = X[indices], y[indices]

            # Optional oversampling
            if self.oversampler_class is not None:
                oversampler = self.oversampler_class(
                    random_state=self.random_state + 10000 + i
                )
                X_sample, y_sample = oversampler.fit_resample(X_sample, y_sample)

            estimator = clone(self.base_estimator)
            estimator.fit(X_sample, y_sample)
            return estimator

        with ThreadPoolExecutor(self.threads) as executor:
            futures = [
                executor.submit(fit_estimator, i) for i in range(self.n_estimators)
            ]
            self.estimators_ = [f.result() for f in futures]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the input samples.
        Args:
            X (np.ndarray): Input samples.
        Returns:
            np.ndarray: Predicted class labels.
        """
        check_is_fitted(self, "estimators_")
        X = check_array(X)

        predictions = np.array([est.predict(X) for est in self.estimators_])
        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )
        return majority_votes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class probabilities for the input samples.
        Args:
            X (np.ndarray): Input samples.
        Returns:
            np.ndarray: Predicted class probabilities.
        """
        check_is_fitted(self, "estimators_")
        X = check_array(X)

        probas = np.mean([est.predict_proba(X) for est in self.estimators_], axis=0)
        return probas
