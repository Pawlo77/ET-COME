"""
This module implements an efficient version of the SMOTE
algorithm for oversampling imbalanced datasets. It uses
a custom KNN callable for neighbor queries.
"""

from typing import Any, Callable, Tuple, Union

import numpy as np
from scipy import sparse
from utils import RANDOM_SEED


# pylint: disable=too-many-ancestors
class EfficientSMOTE:
    """
    An enhanced version of SMOTE that leverages a custom KNN callable for neighbor queries.
    """

    def __init__(
        self,
        knn_callable: Callable[..., Any],
        oversampling_per_step: int,
        k_neighbors: int,
        minority_class: int,
    ) -> None:
        """
        Initializes the EfficientSMOTE instance.

        Args:
            knn_callable (Callable): A callable for performing KNN neighbor queries.
            oversampling_per_step (int): Number of synthetic samples to generate per step.
            k_neighbors (int): Number of neighbors to use for SMOTE.
            minority_class: Minority class to oversample.
        """
        self._knn: Callable[..., Any] = knn_callable
        self._oversampling_per_step = oversampling_per_step
        self._k_neighbors = k_neighbors
        self._minority_class = minority_class

    # pylint: disable=arguments-differ,invalid-name
    def _fit_resample(
        self, X: np.ndarray, y: np.ndarray, indices_to_oversample: np.ndarray
    ) -> Tuple[Union[np.ndarray, sparse.spmatrix], np.ndarray]:
        """
        Resamples the dataset using SMOTE with a custom KNN
        query for synthetic sample generation.

        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Class labels.
            indices_to_oversample (np.ndarray): Indices of samples to be oversampled.

        Returns:
            Tuple[Union[np.ndarray, sparse.spmatrix], np.ndarray]: The
                resampled features and labels.
        """
        X_resampled = X.copy()
        y_resampled = y.copy()

        X_new = []
        y_new = []

        np.random.seed(RANDOM_SEED)

        # Get neighbors for samples to oversample.
        nns = self._knn(
            indices_to_oversample, return_distances_=False
        )[:, : self._k_neighbors]

        # Calculate oversampling per sample.
        oversampling_per_index = self._oversampling_per_step // len(
            indices_to_oversample
        )
        left_overs = self._oversampling_per_step - oversampling_per_index * len(
            indices_to_oversample
        )

        for i, index in enumerate(indices_to_oversample):
            neighbors = nns[i]
            sample = X[index]

            # Choose neighbors for oversampling_per_index
            seed_indices = np.random.choice(neighbors, oversampling_per_index)
            seed_neighbors = X[seed_indices]

            for neighbor in seed_neighbors:
                step = np.random.uniform()
                new_sample = sample + step * (neighbor - sample)
                X_new.append(new_sample)
                y_new.append(self._minority_class)

        # Handle left_overs.
        if left_overs > 0:
            chosen_indices = np.random.choice(
                len(indices_to_oversample), left_overs, replace=False
            )
            for idx in chosen_indices:
                index = indices_to_oversample[idx]
                neighbors = nns[idx]
                sample = X[index]
                neighbor_idx = np.random.choice(neighbors)
                neighbor = X[neighbor_idx]
                step = np.random.uniform()
                new_sample = sample + step * (neighbor - sample)
                X_new.append(new_sample)
                y_new.append(self._minority_class)
        
        if X_new:
            X_resampled = np.vstack([X_resampled, np.array(X_new)])
            y_resampled = np.hstack([y_resampled, np.array(y_new)])

        return X_resampled, y_resampled
