"""
This module implements an efficient version of the SMOTE
algorithm for oversampling imbalanced datasets. It uses
a custom KNN callable for neighbor queries.
"""

from typing import Any, Callable, Tuple, Union

import numpy as np
from imblearn.over_sampling import SMOTE
from scipy import sparse


# pylint: disable=too-many-ancestors
class EfficientSMOTE(SMOTE):
    """
    An enhanced version of SMOTE that leverages a custom KNN callable for neighbor queries.
    """

    def __init__(
        self, knn_callable: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """
        Initializes the EfficientSMOTE instance.

        Args:
            knn_callable (Callable): A callable that performs KNN queries.
            *args: Additional positional arguments to pass to the SMOTE base class.
            **kwargs: Additional keyword arguments to pass to the SMOTE base class.
        """
        super().__init__(*args, **kwargs)
        self._knn: Callable[..., Any] = knn_callable

    # pylint: disable=arguments-differ,invalid-name
    def _fit_resample(
        self, X: np.ndarray, y: np.ndarray, X_to_oversample: np.ndarray
    ) -> Tuple[Union[np.ndarray, sparse.spmatrix], np.ndarray]:
        """
        Resamples the dataset using SMOTE with a custom KNN
        query for synthetic sample generation.

        Args:
            X (np.ndarray): Original feature data.
            y (np.ndarray): Original class labels.
            X_to_oversample (np.ndarray): Samples to be oversampled.

        Returns:
            Tuple[Union[np.ndarray, sparse.spmatrix], np.ndarray]: The
                resampled features and labels.
        """
        X_resampled = []
        y_resampled = []

        for class_sample, n_samples in self.sampling_strategy.items():
            if n_samples == 0:
                continue

            indices = np.flatnonzero(y == class_sample)
            X_class = X_to_oversample[indices]
            nns = self._knn(indices, k=self.k_neighbors, return_distances_=False)

            X_new, y_new = self._make_samples(
                X_class,
                y.dtype,
                class_sample,
                X,
                nns,
                n_samples,
                1.0,
            )

            X_resampled.append(X_new)
            y_resampled.append(y_new)

        return np.vstack(X_resampled), np.hstack(y_resampled)
