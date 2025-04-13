"""
This module implements an efficient version of SMOTE and a custom ensemble classifier for imbalanced data.
It leverages approximate KNN using HNSWlib for fast neighbor searches, applies Edited Nearest Neighbor (ENN)
cleaning, and uses iterative training with oversampling. The classifier is compatible with scikit-learn and
can use any base estimator.
"""

import concurrent.futures
import copy
import logging
import time
from typing import Any, Callable, List, Optional, Tuple, Union

import hnswlib
import numpy as np
import scipy.sparse as sparse
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier


class TimedLogger:
    """
    A context manager for measuring and logging the elapsed time for a block of code.
    """

    def __init__(self, msg: str, logger: logging.Logger, level: int = logging.DEBUG):
        """
        Initializes a TimedLogger instance.

        Args:
            msg (str): The base log message.
            logger (logging.Logger): The logger to use.
            level (int, optional): The logging level. Defaults to logging.DEBUG.
        """
        self.msg = msg
        self.level = level
        self.logger = logger

    def __enter__(self) -> "TimedLogger":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        elapsed_time = time.time() - self.start_time
        self.logger.log(self.level, self.msg + f"... ({elapsed_time:.2f} seconds)")


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

    def _fit_resample(
        self, X: np.ndarray, y: np.ndarray, X_to_oversample: np.ndarray
    ) -> Tuple[Union[np.ndarray, sparse.spmatrix], np.ndarray]:
        """
        Resamples the dataset using SMOTE with a custom KNN query for synthetic sample generation.

        Args:
            X (np.ndarray): Original feature data.
            y (np.ndarray): Original class labels.
            X_to_oversample (np.ndarray): Samples to be oversampled.

        Returns:
            Tuple[Union[np.ndarray, sparse.spmatrix], np.ndarray]: The resampled features and labels.
        """
        X_resampled = []
        y_resampled = []

        for class_sample, n_samples in self.sampling_strategy.items():
            if n_samples == 0:
                continue

            indeces = np.flatnonzero(y == class_sample)
            X_class = X[indeces]
            nns = self._knn(indeces, k=self.k_neighbors, return_distances_=False)

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


class InspireClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom ensemble classifier using iterative training with ENN cleaning and oversampling via SMOTE.
    Base classifiers are trained on folds of the cleaned data, and predictions are aggregated via majority voting.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        base_estimator: ClassifierMixin = DecisionTreeClassifier(),
    ) -> None:
        """
        Initializes the custom ensemble classifier.

        Args:
            n_estimators (int): Total number of base estimators to train.
            base_estimator (ClassifierMixin): The base classifier used by the ensemble.
        """
        self.n_estimators: int = n_estimators
        self.base_estimator: ClassifierMixin = base_estimator

        self._models: List[ClassifierMixin] = []
        self._smote: Optional[EfficientSMOTE] = None

        self._fitted_: bool = False
        self._history: Optional[List[dict]] = None

        self._n_classes = None

        # Cache for full KNN results and associated parameters
        self._full_knn_indices: Optional[np.ndarray] = None
        self._full_knn_distances: Optional[np.ndarray] = None
        self._knn_fitted_: bool = False
        self._knn_val_indices: Optional[np.ndarray] = None
        self._knn_val_distances: Optional[np.ndarray] = None
        self._knn_val_fitted_: bool = False

        # Mapping arrays for translating indices between the cleaned and original datasets.
        self._indices_map: Optional[np.ndarray] = None
        self._removed_mask: Optional[np.ndarray] = None

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

    @property
    def models(self) -> List[ClassifierMixin]:
        """Returns the list of trained base classifiers."""
        return self._models

    @property
    def smote(self) -> Optional[EfficientSMOTE]:
        """Returns the EfficientSMOTE instance."""
        return self._smote

    @property
    def fitted(self) -> bool:
        """Indicates whether the classifier has been fitted."""
        return self._fitted_

    @property
    def history(self) -> List[dict]:
        """Returns the training history, if available."""
        return self._history

    @history.deleter
    def history(self):
        """Deletes the training history."""
        del self._history

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        enn_neighbours: int = 7,
        oversampling_neighbours: int = 10,
        cache_size: int = None,
        step_size: int = 2,
        oversampling_per_step: int = 100,
        bagging_: bool = True,
        approximate_knn_: bool = True,
        remove_outliers_: bool = True,
        perform_oversampling_: bool = True,
        cleanup_: bool = True,
        br_kwargs: dict = dict(br_treshold=0.5),
        bp_kwargs: dict = dict(theta=0.7, tau=0.5, neighbours=3),
        stratified_kwargs: dict = dict(shuffle=True, random_state=42),
        knn_kw: dict = {},
        level: int = logging.INFO,
        n_jobs: int = None,
        save_history_: bool = False,
        minority_class: int = None,
        n_classes: int = None,
    ) -> "CustomEnsembleClassifier":  # noqa: F821
        """
        Fits the ensemble classifier using iterative training with ENN cleaning and oversampling.
        Assumes labels are integers 0, 1, ... n - 1 where n stands for number of classes.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            enn_neighbours (int): Number of neighbours used for ENN cleaning.
            oversampling_neighbours (int): Number of neighbours used to identify border regions.
            cache_size (int): Size of the KNN cache (number of neighbors to cache). If None, it is determined automatically.
            step_size (int): Number of models trained per iteration.
            oversampling_per_step (int): Number of synthetic samples to generate per step.
            bagging_ (bool): Whether to use bagging (via stratified folds).
            approximate_knn_ (bool): Whether to use approximate KNN search.
            remove_outliers_ (bool): Whether to perform ENN cleaning.
            perform_oversampling_ (bool): Whether to perform oversampling via SMOTE.
            cleanup_ (bool): Whether to clean up cached data after training (excluding training history).
            br_kwargs (dict): Parameters for identifying border regions (must include 'br_treshold').
            bp_kwargs (dict): Parameters for identifying bad performance regions (must include 'neighbours').
            stratified_kwargs (dict): Keyword arguments for stratified folding.
            knn_kw (dict): Additional parameters for KNN fitting.
            level (int): Logging level.
            n_jobs (int): Number of jobs for parallel execution. If None, uses all processors.
            save_history_ (bool): Whether to save results from each training step (can be space/time intensive).
            minority_class (int): The minority class label for the dataset. If not
                provided, will be determined automatically.
            n_classes (int): Number of classes in the dataset. If not provided, will be determined automatically.

        Returns:
            CustomEnsembleClassifier: The fitted classifier.

        Raises:
            RuntimeError: If the classifier is already fitted.
            ValueError: If n_estimators is not divisible by step_size or if bp_kwargs does not contain 'neighbours'.
        """
        if self._fitted_:
            raise RuntimeError("Classifier already fitted.")

        if self.n_estimators % step_size != 0:
            raise ValueError("n_estimators must be divisible by step_size")

        if "neighbours" not in bp_kwargs:
            raise ValueError("bp_kwargs must contain 'neighbours' key")
        if save_history_:
            self._history = []

        self._logger.setLevel(level)
        self._logger.debug("Starting fit process.")

        if cache_size is None:
            if remove_outliers_:
                cache_size = (
                    max(enn_neighbours, oversampling_neighbours) + enn_neighbours
                )
            else:
                cache_size = max(enn_neighbours, oversampling_neighbours)

        # Step 1: Fit and cache the KNN index for the training set.
        with TimedLogger("Fitting KNN index", logger=self._logger, level=logging.INFO):
            self._fit_knn(
                X_train=X, approximate_knn_=approximate_knn_, k=cache_size, **knn_kw
            )

        # Step 2: Perform ENN cleaning on the training data.
        if remove_outliers_:
            with TimedLogger(
                "Removing outliers", logger=self._logger, level=logging.INFO
            ):
                X_clean, y_clean = self._perform_enn(X, y, k=enn_neighbours)
            if save_history_:
                self._save_history_entry("ENN", removed_mask=self._removed_mask)
        else:
            X_clean, y_clean = X, y
        indeces = np.arange(len(X_clean))

        # Step 0: Identify minority class
        if minority_class is None:
            unique, counts = np.unique(y_clean, return_counts=True)
            minority_class = unique[np.argmin(counts)]
            self._logger.info(f"Identified minority class: {minority_class}")
        val_minority_mask = y_val == minority_class
        minority_mask = y_clean == minority_class

        if n_classes is None:
            n_classes = len(np.unique(y_clean))
            self._logger.info(f"Identified number of classes: {n_classes}")
        self._n_classes = n_classes

        # Step 1 (Validation): Fit and cache the KNN index for the validation set.
        with TimedLogger(
            "Fitting KNN index for validation set",
            logger=self._logger,
            level=logging.INFO,
        ):
            # fit just between validation set and minority class in training set
            self._fit_knn_val(
                X_train=X_clean[minority_mask],
                X_val=X_val,
                approximate_knn_=approximate_knn_,
                k=bp_kwargs["neighbours"],
                **knn_kw,
            )

        # Step 3: Identify border regions for oversampling.
        border_mask = None
        if perform_oversampling_:
            with TimedLogger(
                "Identifying border regions", logger=self._logger, level=logging.INFO
            ):
                border_mask = self._identify_br(
                    indeces,
                    y_clean,
                    k=oversampling_neighbours,
                    minority_class=minority_class,
                    **br_kwargs,
                )
            if save_history_:
                self._save_history_entry("BR", border_mask=border_mask)

        # Step 4: Split the cleaned training data into folds.
        if bagging_:
            skf = StratifiedKFold(n_splits=self.n_estimators, **stratified_kwargs)
            folds = list(skf.split(X_clean, y_clean))
            if save_history_:
                self._save_history_entry("Bagging", folds=folds)
        else:
            indices = np.arange(len(X_clean))
            folds = [(indices, indices) for _ in range(self.n_estimators)]

        # initialize smote and training cache
        cached_class_preds = None
        self._smote = EfficientSMOTE(
            knn_callable=self._knn,
            sampling_strategy={minority_class: oversampling_per_step},
            k_neighbors=oversampling_neighbours,
        )

        # Step 5: Iterative training using parallel execution.
        with concurrent.futures.ThreadPoolExecutor(n_jobs) as executor:
            for i in range(self.n_estimators // step_size):
                X_synth, y_synth = None, None

                if i > 0:
                    with TimedLogger(
                        f"Performing adaptive optimizations (step {i})",
                        logger=self._logger,
                        level=logging.INFO,
                    ):
                        batch_models = self._models[-step_size:]
                        preds, cached_class_preds = self._evaluate_batch(
                            X_val,
                            y_val,
                            batch_models,
                            cached_class_preds=cached_class_preds,
                        )
                        if save_history_:
                            self._save_history_entry(
                                f"Batch__{i}__preds",
                                preds=preds,
                                cached_class_preds=cached_class_preds,
                            )
                        if perform_oversampling_:
                            if save_history_:
                                bp_mask, bp_history = self._identify_bp(
                                    X_clean,
                                    y_val,
                                    minority_mask=minority_mask,
                                    val_minority_mask=val_minority_mask,
                                    class_preds=cached_class_preds,
                                    return_history_=True,
                                    **bp_kwargs,
                                )
                            else:
                                bp_history = {}
                                bp_mask = self._identify_bp(
                                    X_clean,
                                    y_val,
                                    minority_mask=minority_mask,
                                    val_minority_mask=val_minority_mask,
                                    class_preds=cached_class_preds,
                                    return_history_=False,
                                    **bp_kwargs,
                                )

                            # calculate regions to oversample
                            oversampling_indeces = indeces[border_mask & bp_mask]
                            X_to_oversample = X_clean[oversampling_indeces]

                            if len(X_to_oversample) > 0:
                                X_synth, y_synth = self._perform_smote(
                                    X_clean, y_clean, X_to_oversample
                                )
                            else:
                                self._logger.warning("No regions to oversample found.")

                            if save_history_:
                                self._save_history_entry(
                                    f"Batch__{i}__oversampling",
                                    oversampling_indeces=oversampling_indeces,
                                    X_synth=X_synth,
                                    y_synth=y_synth,
                                    **bp_history,
                                )

                # train next models batch
                with TimedLogger(
                    f"Training models (step {i})",
                    logger=self._logger,
                    level=logging.INFO,
                ):
                    futures = [
                        executor.submit(
                            InspireClassifier._train_model,
                            clf=copy.deepcopy(self.base_estimator),
                            X_clean=X_clean,
                            y_clean=y_clean,
                            folds=folds,
                            fold_idx=i * step_size + j,
                            X_synth=X_synth,
                            y_synth=y_synth,
                        )
                        for j in range(step_size)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        self._models.append(future.result())

        # remove all cache and data no longer needed
        if cleanup_:
            del self._full_knn_indices
            del self._full_knn_distances
            del self._knn_val_indices
            del self._knn_val_distances
            del self._indices_map
            del self._removed_mask
            del self._smote

        self._fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray,
        level: int = logging.INFO,
        n_jobs: int = None,
        soft_: bool = False,
    ) -> np.ndarray:
        """
        Predicts class labels for the given features using majority voting across the ensemble.

        Args:
            X (np.ndarray): Input feature data.
            level (int): Logging level.
            n_jobs (int, optional): Number of parallel jobs. If None, uses all processors.
            soft_ (bool): If True, applies soft voting; otherwise, applies hard voting.

        Returns:
            np.ndarray: The predicted class labels.
        """
        self._logger.setLevel(level)
        with TimedLogger(
            "Performing predictions", logger=self._logger, level=logging.INFO
        ):
            with concurrent.futures.ThreadPoolExecutor(n_jobs) as executor:
                preds = np.array(
                    list(executor.map(lambda clf: clf.predict(X), self._models))
                )
            return self._vote(preds=preds, soft_=soft_)

    def _vote(self, preds: np.ndarray, soft_: bool = False) -> np.ndarray:
        """
        Apply soft voting to the predictions.

        Args:
            preds (np.ndarray): The predictions from the ensemble.
            soft_ (bool): If True, applies soft voting; otherwise, applies hard voting.

        Returns:
            np.ndarray: The final predictions based on soft voting.
        """
        if soft_:
            return np.apply_along_axis(
                lambda x: np.bincount(x, minlength=self._n_classes) / preds.shape[0],
                axis=0,
                arr=preds,
            )
        return np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self._n_classes).argmax(),
            axis=0,
            arr=preds,
        )

    def _save_history_entry(self, name: str, **dt: dict) -> None:
        """
        Saves a training history entry.

        Args:
            name (str): Name for the history entry.
            **dt: Additional key-value pairs to store.
        """
        self._history.append({"name": name, **dt})

    def _fit_knn_val(self, **kwargs) -> None:
        """
        Fits a KNN index for the validation set and caches the results.

        Raises:
            RuntimeError: If the validation KNN index is already fitted.
        """
        if self._knn_val_fitted_:
            raise RuntimeError("KNN index for validation set already fitted.")
        self._logger.debug("Fitting KNN index for validation set.")
        self._knn_val_indices, self._knn_val_distances = self._fit_knn_raw(**kwargs)
        self._knn_val_fitted_ = True

    def _fit_knn(self, X_train: np.ndarray, k: int = 5, **kwargs: dict) -> None:
        """
        Fits a KNN index for the training set and caches the results.

        Args:
            X_train (np.ndarray): Training data.
            k (int): Number of neighbors to cache.
            **kwargs: Additional keyword arguments for KNN fitting.

        Raises:
            RuntimeError: If the training KNN index is already fitted.
        """
        if self._knn_fitted_:
            raise RuntimeError("KNN index already fitted.")
        self._logger.debug("Fitting KNN index.")
        indeces, distances = self._fit_knn_raw(
            X_train=X_train, X_val=X_train, k=k + 1, **kwargs
        )
        # Do not store the first index as it is the same point.
        self._full_knn_indices, self._full_knn_distances = (
            indeces[:, 1:],
            distances[:, 1:],
        )
        self._knn_fitted_ = True

    def _fit_knn_raw(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        approximate_knn_: bool = False,
        k: int = 5,
        ef: int = 500,
        ef_construction: int = 200,
        M: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits a KNN index using either approximate or exact search and returns the indices and distances.

        Args:
            X_train (np.ndarray): Training data to index.
            X_val (np.ndarray): Data to query the index.
            approximate_knn_ (bool): Whether to use approximate KNN.
            k (int, optional): Number of neighbors to retrieve. Defaults to 5.
            ef (int, optional): HNSWlib ef parameter. Defaults to 500.
            ef_construction (int, optional): HNSWlib construction parameter. Defaults to 200.
            M (int, optional): HNSWlib M parameter. Defaults to 16.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The KNN indices and distances for the queried data.
        """
        if approximate_knn_:
            self._logger.debug("Using approximate KNN.")
            dim = X_train.shape[1]
            n_elements = X_train.shape[0]
            max_elements = n_elements * k + 1
            knn_index = hnswlib.Index(space="l2", dim=dim)
            knn_index.init_index(
                max_elements=max_elements, ef_construction=ef_construction, M=M
            )
            knn_index.add_items(X_train, np.arange(n_elements))
            knn_index.set_ef(ef)
            indices, distances = knn_index.knn_query(X_val, k=k)
        else:
            self._logger.debug("Using exact KNN.")
            neigh = NearestNeighbors(n_neighbors=k, metric="euclidean")
            neigh.fit(X_train)
            distances, indices = neigh.kneighbors(X_val)
        return indices, distances

    def _knn_val(
        self, indeces: np.ndarray, k: int = 5, return_distances_: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Queries the validation KNN index and returns neighbor information.

        Args:
            indeces (np.ndarray): Indices in the validation set to query.
            k (int, optional): Number of neighbors to return. Defaults to 5.
            return_distances_ (bool, optional): If True, also returns neighbor distances. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
                A tuple (neighbor_indices, neighbor_distances) if return_distances_ is True,
                or just the neighbor indices.

        Raises:
            RuntimeError: If the validation KNN index is not yet fitted.
            ValueError: If k exceeds the number of cached neighbors.
        """
        if not self._knn_val_fitted_:
            raise RuntimeError("KNN val index not fitted. Call _fit_knn_val first.")
        if k > self._knn_val_indices.shape[1]:
            raise ValueError("k exceeds the number of cached neighbors.")
        self._logger.debug("Using cached validation KNN results.")
        neighbours_indeces = self._knn_val_indices[indeces]
        if not return_distances_:
            return neighbours_indeces[:, :k]
        neighbours_distances = self._knn_val_distances[indeces]
        return neighbours_indeces[:, :k], neighbours_distances[:, :k]

    def _knn(
        self, indeces: np.ndarray, k: int = 5, return_distances_: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Queries the training KNN index for the given indices and returns neighbor information.

        Args:
            indeces (np.ndarray): Indices in the training set to query.
            k (int, optional): Number of neighbors to return. Defaults to 5.
            return_distances_ (bool, optional): If True, also returns neighbor distances. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
                A tuple (neighbor_indices, neighbor_distances) if return_distances_ is True,
                or just the neighbor indices.

        Raises:
            RuntimeError: If the training KNN index is not yet fitted.
            ValueError: If k exceeds the number of cached neighbors.
        """
        if not self._knn_fitted_:
            raise RuntimeError("KNN index not fitted. Call _fit_knn first.")
        if k > self._full_knn_indices.shape[1]:
            raise ValueError("k exceeds the number of cached neighbors.")
        self._logger.debug("Using cached training KNN results.")
        indeces = self._translate_indices(indeces, direction="NO")
        neighbours_indeces = self._full_knn_indices[indeces]
        if self._removed_mask is not None:
            mask = self._removed_mask[neighbours_indeces]
            filtered = []
            for row, row_mask in zip(neighbours_indeces, mask):
                valid = row[~row_mask]
                if valid.size < k:
                    raise ValueError(
                        "Less than k valid neighbors available in at least one row."
                    )
                filtered.append(valid[:k])
            neighbours_indeces = np.array(filtered)
        else:
            neighbours_indeces = neighbours_indeces[:, :k]
        neighbours_indeces = self._translate_indices(neighbours_indeces, direction="ON")
        if not return_distances_:
            return neighbours_indeces
        neighbours_distances = self._full_knn_distances[indeces]
        filtered = []
        # If a removed mask was used, filter distances similarly.
        if self._removed_mask is not None:
            mask = self._removed_mask[neighbours_indeces]
            for row, row_mask in zip(neighbours_distances, mask):
                valid = row[~row_mask]
                filtered.append(valid[:k])
            neighbours_distances = np.array(filtered)
        else:
            neighbours_distances = neighbours_distances[:, :k]
        return neighbours_indeces, neighbours_distances

    def _translate_indices(
        self, indices: np.ndarray, direction: str = "NO"
    ) -> np.ndarray:
        """
        Translates indices between the cleaned and original data representations.

        Args:
            indices (np.ndarray): The indices to translate.
            direction (str): Translation direction; "NO" translates new->old, "ON" translates old->new.

        Returns:
            np.ndarray: The translated indices.

        Raises:
            ValueError: If an invalid translation direction is provided.
        """
        self._logger.debug("Translating indices.")
        if direction not in ["NO", "ON"]:
            raise ValueError("Invalid direction. Use 'NO' or 'ON'.")
        if self._indices_map is None:
            return indices
        if direction == "NO":
            return self._indices_map[indices, 0]
        return self._indices_map[indices, 1]

    def _perform_enn(
        self, X: np.ndarray, y: np.ndarray, k: int = 5, min_allowed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs Edited Nearest Neighbor (ENN) cleaning to remove potential outliers.

        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Class labels.
            k (int, optional): Number of neighbors to consider for ENN. Defaults to 5.
            min_allowed (int, optional): Minimum number of matching neighbors required. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The cleaned features and corresponding labels.
        """
        self._logger.debug("Performing ENN cleaning.")
        indices = np.arange(len(X))
        indices = self._knn(indices, k=k, return_distances_=False)
        neighbor_labels = np.array(y[indices.ravel()]).reshape(indices.shape)
        keep_mask = (
            np.sum(neighbor_labels == np.array(y)[:, np.newaxis], axis=1) >= min_allowed
        )
        X_clean = X[keep_mask]
        y_clean = y[keep_mask]
        self._logger.debug(f"Removed {len(X) - len(X_clean)} entries.")
        new_old_map = -np.ones(X.shape[0], dtype=int)
        old_new_map = -np.ones(X.shape[0], dtype=int)
        new_old_map[: X_clean.shape[0]] = np.flatnonzero(keep_mask)
        old_new_map[keep_mask] = np.arange(X_clean.shape[0])
        self._indices_map = np.column_stack((new_old_map, old_new_map))
        self._removed_mask = ~keep_mask
        return X_clean, y_clean

    def _evaluate_batch(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_models: List[ClassifierMixin],
        cached_class_preds: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates a batch of models on validation data and aggregates their predictions.

        Args:
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            batch_models (List[ClassifierMixin]): The list of models to evaluate.
            cached_class_preds (Optional[np.ndarray]): Previously cached predictions, if any.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the majority vote predictions and all aggregated predictions.
        """
        self._logger.debug("Evaluating batch models (%d).", len(batch_models))
        batch_class_preds = np.array([clf.predict(X_val) for clf in batch_models])
        self._logger.debug(f"Cached preds: {cached_class_preds}")
        if cached_class_preds is not None:
            all_class_preds = np.concatenate(
                [cached_class_preds, batch_class_preds], axis=0
            )
        else:
            all_class_preds = batch_class_preds
        majority_vote = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=2).argmax(), axis=0, arr=all_class_preds
        )
        return majority_vote, all_class_preds

    def _identify_bp(
        self,
        X_clean: np.ndarray,
        y_val: np.ndarray,
        class_preds: np.ndarray,
        minority_mask: np.ndarray,
        val_minority_mask: np.ndarray,
        theta: float = 0.7,
        tau: float = 0.5,
        neighbours: int = 3,
        return_history_: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Identifies regions of poor performance (Bad Performance - BP) in the validation data.

        Warning:
            Implementation supports just binary classification.

        Args:
            X_clean (np.ndarray): Cleaned training samples.
            y_val (np.ndarray): True labels for the validation data.
            class_preds (np.ndarray): Predictions from the ensemble (for each model in ensable,
                2D array of stacked predictions).
            val_minority_mask (np.ndarray): Mask for the minority class in the validation data.
            minority_class (int): The minority class label.
            theta (float): Confidence threshold.
            tau (float): Misclassification proportion threshold.
            neighbours (int): Number of neighbors to use for pinpointing BP regions.
            return_history_ (bool): If True, returns the history of BP regions.

        Returns:
            np.ndarray: A boolean mask indicating the BP regions.
        """
        self._logger.debug("Identifying BP regions.")

        # Calculate the mean prediction probabilities for each class
        # for validation data where true label is minority class
        votes = self._vote(preds=class_preds, soft_=True)

        confidence_score = np.max(votes, axis=0)
        y_pred = np.argmax(votes, axis=0)

        # model is unsure of its prediction
        condidence_mask = confidence_score < theta
        # model is wrong
        misclass_mask = y_pred != y_val

        val_bp_mask = condidence_mask | misclass_mask
        val_bp_indeces = np.flatnonzero(val_bp_mask)

        minority_train_bp_indeces = self._knn_val(
            val_bp_indeces, k=neighbours, return_distances_=False
        ).flatten()

        # to be optimized with indeces transalation
        train_bp_mask = np.zeros(X_clean.shape[0], dtype=bool)
        minority_train_bp_minority = train_bp_mask[minority_mask]
        minority_train_bp_minority[minority_train_bp_indeces] = True
        train_bp_mask[minority_mask] = minority_train_bp_minority

        if not return_history_:
            return train_bp_mask

        return train_bp_mask, {
            "condidence_mask": condidence_mask,
            "y_pred": y_pred,
            "misclass_mask": misclass_mask,
            "val_bp_mask": val_bp_mask,
            "bp_mask": train_bp_mask,
            "minority_train_bp_indeces": minority_train_bp_indeces,
            "minority_mask": minority_mask,
        }

    def _identify_br(
        self,
        indeces: np.ndarray,
        y_clean: np.ndarray,
        minority_class: int,
        k: int = 5,
        br_treshold: float = 0.5,
    ) -> np.ndarray:
        """
        Identifies borderline regions (BR) for oversampling based on neighbor inconsistencies.

        Args:
            indeces (np.ndarray): Indices of samples to query.
            y_clean (np.ndarray): Cleaned class labels.
            minority_class (int): The minority class label.
            k (int, optional): Number of neighbors to query. Defaults to 5.
            br_treshold (float, optional): Threshold for determining borderline regions
                (e.g. 0.5 means at least 50% of neighbors are of a different class). Defaults to 0.5.

        Returns:
            np.ndarray: A boolean mask indicating borderline regions.
        """
        self._logger.debug("Identifying BR regions.")
        neighbours_indices = self._knn(indeces, k=k, return_distances_=False)
        neighbor_labels = np.array(y_clean[neighbours_indices.ravel()]).reshape(
            neighbours_indices.shape
        )

        br_mask = (
            np.sum(neighbor_labels == y_clean[:, np.newaxis], axis=1)
            <= br_treshold * neighbor_labels.shape[1]
        ) & (y_clean == minority_class)
        return br_mask

    def _perform_smote(
        self, X_clean: np.ndarray, y_clean: np.ndarray, X_to_oversample: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies SMOTE to generate synthetic samples for the minority class.

        Args:
            X_clean (np.ndarray): Cleaned feature data.
            y_clean (np.ndarray): Cleaned labels.
            X_to_oversample (np.ndarray): Samples selected for oversampling.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The synthetic features and labels generated by SMOTE.
        """
        return self._smote._fit_resample(X_clean, y_clean, X_to_oversample)

    @staticmethod
    def _train_model(
        clf: BaseEstimator,
        X_clean: np.ndarray,
        y_clean: np.ndarray,
        folds: list,
        fold_idx: int,
        X_synth: np.ndarray,
        y_synth: np.ndarray,
    ) -> BaseEstimator:
        """
        Trains a single instance of the base classifier using training data from a cross-validation fold
        and optional synthetic data.

        Args:
            clf (BaseEstimator): The classifier to train.
            X_clean (np.ndarray): Cleaned training features.
            y_clean (np.ndarray): Cleaned training labels.
            folds (list): A list of fold splits; each element is a (train_indices, val_indices) tuple.
            fold_idx (int): The fold index to exclude from training.
            X_synth (np.ndarray): Synthetic feature data for oversampling (optional).
            y_synth (np.ndarray): Synthetic labels corresponding to X_synth (optional).

        Returns:
            BaseEstimator: A trained instance (deep-copied) of the base classifier.
        """
        train_idx = np.concatenate(
            [folds[idx][1] for idx in range(len(folds)) if idx != fold_idx]
        )
        X_train = X_clean[train_idx]
        y_train = y_clean[train_idx]
        if X_synth is not None:
            X_train = np.vstack([X_train, X_synth])
            y_train = np.hstack([y_train, y_synth])
        clf.fit(X_train, y_train)
        return clf
