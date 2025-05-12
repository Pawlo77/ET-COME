"""
A custom ensemble classifier for imbalanced data.
It leverages approximate KNN using HNSWlib for fast neighbor searches, applies Edited Nearest Neighbor (ENN)
cleaning, and uses iterative training with oversampling. The classifier is compatible with scikit-learn and
can use any base estimator.
"""

import concurrent.futures
import logging
from enum import Enum
from typing import List, Optional, Tuple, Union

import hnswlib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .bagging_classifier import BaggingClassifier
from .efficient_smote import EfficientSMOTE
from .utils import TimedLogger, create_logger

logger = create_logger(__name__)


class Direction(Enum):
    """
    Enum for direction of index translation.
    """

    NEW_OLD = 1  # new -> old
    OLD_NEW = 2  # old -> new


class MergingStrategy(Enum):
    """
    Enum for merging strategies.
    """

    AND = 1
    OR = 2


class InspireClassifier(BaggingClassifier):
    """
    A custom ensemble classifier using iterative
    training with ENN cleaning and oversampling via SMOTE-BORDERLINE.
    Base classifiers are trained on the cleaned data
    in bagging terminology.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        bootstrap: bool = True,
        random_state: int = None,
        threads: int = 4,
        minority_class: int = None,
        #
        enn_neighbors: int = 7,
        enn_min_matching_neighbors: int = 3,
        oversampling_neighbors: int = 10,
        val_to_train_neighbors: int = 3,
        knn_kw: dict = {},
        approximate_knn_: bool = True,
        #
        step_size: int = 4,
        cache_size: int = None,
        #
        oversampling_per_step: int = None,
        oversampling_ratio: float = None,
        #
        br_threshold: float = 0.5,
        bp_theta: float = 0.7,
        mask_merging_strategy: MergingStrategy = MergingStrategy.AND,
        #
        verbose_: bool = False,
        remove_outliers_: bool = True,
    ) -> None:
        """
        Initializes the custom ensemble classifier.

        Args:
            base_estimator (BaseEstimator): The base estimator to fit on random subsets of the data.
            n_estimators (int): The number of base estimators in the ensemble.
            max_samples (float): The fraction of samples to draw from X to train each base estimator.
            bootstrap (bool): Whether samples are drawn with replacement.
            random_state (int): Random seed for reproducibility.
            threads (int): Number of threads to use for parallel processing.

            enn_neighbors (int): Number of neighbors for Edited Nearest Neighbor (ENN) cleaning.
            enn_min_matching_neighbors (int): Minimum number of matching neighbors for ENN cleaning.
            oversampling_neighbors (int): Number of neighbors for oversampling.
            val_to_train_neighbors (int): Number of neighbors for validation to training set mapping.
            knn_kw (dict): Keyword arguments for KNN.

            step_size (int): Number of models to train in each step.
            cache_size (int): Size of the KNN cache.

            oversampling_per_step (int): Number of samples to oversample in each step.
            oversampling_ratio (float): Ratio of samples to oversample based on the minority class.
                If provided, it will oversample ratio * n_minority samples (valid, ie after ENN cleaning).

            br_threshold (float): Threshold for identifying border regions.
            bp_theta (float): Confidence threshold for identifying bad performance regions.

            mask_merging_strategy (MergingStrategy): Strategy for merging masks.
            verbose_ (bool): Whether to print progress.
            approximate_knn_ (bool): Whether to use approximate KNN.
            remove_outliers_ (bool): Whether to remove outliers using ENN cleaning.
        """
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            random_state=random_state,
            threads=threads,
        )

        if oversampling_per_step and oversampling_ratio:
            raise ValueError(
                "Specify only one of 'oversampling_per_step' or 'oversampling_ratio', not both."
            )
        elif not (oversampling_per_step or oversampling_ratio):
            raise ValueError(
                "Specify only one of 'oversampling_per_step' or 'oversampling_ratio'."
            )
        if "k" in knn_kw:
            raise ValueError("'k' is not allowed in knn_kw and bp_kwargs.")
        if (
            enn_neighbors <= 0
            or enn_min_matching_neighbors <= 0
            or oversampling_neighbors <= 0
            or val_to_train_neighbors <= 0
            or step_size <= 0
        ):
            raise ValueError(
                "enn_neighbors, enn_min_matching_neighbors, oversampling_neighbors, "
                "val_to_train_neighbors, and step_size must be greater than 0."
            )
        if cache_size is not None and cache_size < max(
            enn_neighbors, oversampling_neighbors
        ):
            raise ValueError(
                "cache_size must be greater than or equal to max(enn_neighbors, oversampling_neighbors)."
            )
        if n_estimators % step_size != 0:
            raise ValueError("n_estimators must be divisible by step_size")
        if enn_min_matching_neighbors > enn_neighbors:
            raise ValueError(
                "enn_min_matching_neighbors must be less than or equal to enn_neighbors."
            )

        self.minority_class = minority_class

        self.enn_neighbors = enn_neighbors  # data cleaning
        self.enn_min_matching_neighbors = enn_min_matching_neighbors  # data cleaning
        self.oversampling_neighbors = oversampling_neighbors  # oversampling
        self.val_to_train_neighbors = (
            val_to_train_neighbors  # translation between X_val and X
        )
        self.knn_kw = knn_kw
        self.approximate_knn_ = approximate_knn_

        self.step_size = step_size
        if cache_size is None:
            if remove_outliers_:
                cache_size = max(enn_neighbors, oversampling_neighbors) + enn_neighbors
            else:
                cache_size = max(enn_neighbors, oversampling_neighbors)
        self.cache_size = cache_size

        self.oversampling_per_step = oversampling_per_step
        self.oversampling_ratio = oversampling_ratio

        self.br_threshold = br_threshold
        self.bp_theta = bp_theta
        self.mask_merging_strategy = mask_merging_strategy

        self.verbose_ = verbose_
        self.remove_outliers_ = remove_outliers_

        # Cache for full KNN results and associated parameters
        self._full_knn_indices: Optional[np.ndarray] = None
        self._full_knn_distances: Optional[np.ndarray] = None
        self._knn_val_indices: Optional[np.ndarray] = None
        self._knn_val_distances: Optional[np.ndarray] = None
        self._knn_val_fitted_: bool = False
        self._knn_fitted_ = False

        # Mapping arrays for translating indices between the cleaned and original datasets.
        self._indices_translation_map: Optional[np.ndarray] = None
        self._removed_mask: Optional[np.ndarray] = None

        self._smote: Optional[EfficientSMOTE] = None
        self._history: List[dict] | None = None

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
        remove_cache_: bool = True,
        save_history_: bool = False,
    ) -> "InspireClassifier":
        """
        Fits the model.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            remove_cache_ (bool): Whether to clean up cached data after training (excluding training history).
            save_history_ (bool): Whether to save results from each training step (can be space/time intensive).
        Returns:
            InspireClassifier: The fitted classifier instance.
        Raises:
            RuntimeError: If the classifier is already fitted.
            ValueError: If n_estimators is not divisible by step_size or if bp_kwargs does not contain 'neighbors'.
        """
        if self._fitted_:
            raise ValueError("This classifier instance is already fitted.")

        X, y = check_X_y(X, y)
        indices = np.arange(len(X))
        if save_history_:
            self._history = []

        logger.debug("Starting fit process.")
        # Step 1 (Train): Fit and cache the KNN index for the training set.
        with TimedLogger("Fitting KNN index", logger=logger, level=logging.INFO):
            self._fit_knn(X_train=X)

        # Step 2: Perform ENN cleaning on the training data.
        if self.remove_outliers_:
            with TimedLogger("Removing outliers", logger=logger, level=logging.INFO):
                X, y = self._perform_enn(indices, X, y)
            if save_history_:
                self._save_history_entry("ENN", removed_mask=self._removed_mask)

        self.classes_, counts = np.unique(y, return_counts=True)
        ratios = self.get_class_ratio(y)

        # Step 3: Identify minority class
        if self.minority_class is None:
            self.minority_class = self.classes_[np.argmin(counts)]
            logger.debug("Identified minority class: %s", self.minority_class)

        minority_mask = y == self.minority_class
        minority_mask_val = y_val == self.minority_class

        # Step 1 (Validation): Fit and cache the KNN index for the validation set.
        with TimedLogger(
            "Fitting KNN index for validation set",
            logger=logger,
            level=logging.INFO,
        ):
            # fit just between validation set and minority class in training set
            self._fit_knn_val(X=X[minority_mask], X_query=X_val, **self.knn_kw)

        # Step 3: Identify border regions for oversampling.
        with TimedLogger(
            "Identifying border regions", logger=logger, level=logging.INFO
        ):
            border_mask = self._identify_br(
                indices,
                y,
                **self._br_kwargs,
            )
        if save_history_:
            self._save_history_entry("BR", border_mask=border_mask)

        # Determine oversampling_per_step based on the oversampling ratio.
        if not self.oversampling_per_step and self.oversampling_ratio:
            n_minority = sum(minority_mask)
            oversampling_per_step = int(n_minority * self.oversampling_ratio)
            logger.debug("Oversampling per step: %d", oversampling_per_step)

        # initialize smote and training cache
        self._smote = EfficientSMOTE(
            knn_callable=self._query_knn,
            sampling_strategy={self.minority_class: oversampling_per_step},
            k_neighbors=self.oversampling_neighbors,
        )

        # Step 5: Iterative training using parallel execution.
        gen = range(self.n_estimators // self.step_size)
        if self.verbose_:
            gen = tqdm(
                gen,
                desc="Training progress",
            )
        with concurrent.futures.ThreadPoolExecutor(self.threads) as executor:
            for step in gen:
                # synthetic samples for the current step
                X_synth, y_synth = None, None

                if step > 0:  # perform oversampling
                    pass

                # train next models batch
                with TimedLogger(
                    f"Training models (step {step})",
                    logger=logger,
                    level=logging.INFO,
                ):
                    futures = [
                        executor.submit(
                            InspireClassifier._train_model,
                            idx=i,
                            X=X,
                            y=y,
                            ratios=ratios,
                            X_synth=X_synth,
                            y_synth=y_synth,
                        )
                        for i in range(self.step_size)
                    ]
                    self.estimators_.extend([f.result() for f in futures])

        # remove all cache and data no longer needed
        if remove_cache_:
            del self._full_knn_indices
            del self._full_knn_distances
            del self._knn_val_indices
            del self._knn_val_distances
            del self._indices_translation_map
            del self._removed_mask
            del self._smote

        self._fitted_ = True
        return self

    def _save_history_entry(self, name: str, **dt: dict) -> None:
        """
        Saves a training history entry.

        Args:
            name (str): Name for the history entry.
            **dt: Additional key-value pairs to store.
        """
        self._history.append({"name": name, **dt})

    def _fit_knn_val(self, **knn_kw) -> None:
        """
        Fits a KNN index for the validation set and caches the results.

        Args:
            **knn_kw: Keyword arguments for KNN fitting.
        Raises:
            RuntimeError: If the validation KNN index is already fitted.
        """
        if self._knn_val_fitted_:
            raise RuntimeError("KNN index for validation set already fitted.")
        logger.debug("Fitting KNN index for validation set.")

        self._knn_val_indices, self._knn_val_distances = self._fit_knn_raw(
            k=self.val_to_train_neighbors, **knn_kw
        )
        self._knn_val_fitted_ = True

    def _fit_knn(self, X: np.ndarray, **knn_kw) -> None:
        """
        Fits a KNN index for the training set and caches the results.

        Args:
            X (np.ndarray): Training data.
            **knn_kw: Keyword arguments for KNN fitting.
        Raises:
            RuntimeError: If the training KNN index is already fitted.
        """
        if self._knn_fitted_:
            raise RuntimeError("KNN index already fitted.")
        logger.debug("Fitting KNN index.")

        indices, distances = self._fit_knn_raw(
            X=X, X_val=X, k=self.cache_size + 1, **knn_kw
        )
        # Do not store the first index as it is the same observation.
        self._full_knn_indices, self._full_knn_distances = (
            indices[:, 1:],
            distances[:, 1:],
        )

        self._knn_fitted_ = True

    def _fit_knn_raw(
        self,
        X: np.ndarray,
        X_query: np.ndarray,
        k: int = 5,
        ef: int = 500,
        ef_construction: int = 200,
        M: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits a KNN index using either approximate or exact search and returns the indices and distances.

        Args:
            X (np.ndarray): Training data to index.
            X_query (np.ndarray): Data to query the index.
            approximate_knn_ (bool): Whether to use approximate KNN.
            k (int, optional): Number of neighbors to retrieve. Defaults to 5.
            ef (int, optional): HNSWlib ef parameter. Defaults to 500.
            ef_construction (int, optional): HNSWlib construction parameter. Defaults to 200.
            M (int, optional): HNSWlib M parameter. Defaults to 16.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The KNN indices and distances for the queried data.
        """
        if self.approximate_knn_:
            logger.debug("Using approximate KNN.")

            n_elements = X.shape[0]
            max_elements = n_elements * k + 1  # 1 for safety
            knn_index = hnswlib.Index(space="l2", dim=X.shape[1])
            knn_index.init_index(
                max_elements=max_elements, ef_construction=ef_construction, M=M
            )
            knn_index.add_items(X, np.arange(n_elements))
            knn_index.set_ef(ef)

            indices, distances = knn_index.knn_query(X_query, k=k)

        else:
            logger.debug("Using exact KNN.")

            neigh = NearestNeighbors(n_neighbors=k, metric="euclidean")
            neigh.fit(X)

            distances, indices = neigh.kneighbors(X_query)

        return indices, distances

    def _translate_val_to_train(
        self, indices: np.ndarray, return_distances_: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Queries the validation KNN index and returns neighbor information.

        Args:
            indices (np.ndarray): Indices in the validation set to query.
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
        logger.debug("Using cached validation KNN results.")

        neighbors_indices = self._knn_val_indices[indices]
        if not return_distances_:
            return neighbors_indices[:, : self.val_to_train_neighbors]

        neighbors_distances = self._knn_val_distances[indices]
        return (
            neighbors_indices[:, : self.val_to_train_neighbors],
            neighbors_distances[:, : self.val_to_train_neighbors],
        )

    def _query_knn(
        self, indices: np.ndarray, return_distances_: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Queries the training KNN index for the given indices and returns neighbor information.

        Args:
            indices (np.ndarray): Indices in the training set to query.
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
        logger.debug("Using cached training KNN results.")

        indices = self._translate_indices(indices, direction=Direction.NEW_OLD)
        neighbors_indices = self._full_knn_indices[indices]

        if self._removed_mask is not None:  # filter out removed samples
            mask = self._removed_mask[neighbors_indices]
            filtered = []
            for row, row_mask in zip(neighbors_indices, mask):
                valid = row[~row_mask]
                if valid.size < self.oversampling_neighbors:
                    raise ValueError(
                        "Less than k valid neighbors available in at least one row."
                    )
                filtered.append(valid[: self.oversampling_neighbors])
            neighbors_indices = np.array(filtered)
        else:
            neighbors_indices = neighbors_indices[:, : self.oversampling_neighbors]

        neighbors_indices = self._translate_indices(
            neighbors_indices, direction=Direction.OLD_NEW
        )
        if not return_distances_:
            return neighbors_indices
        return neighbors_indices, self._full_knn_distances[neighbors_indices]

    def _translate_indices(
        self, indices: np.ndarray, direction: Direction
    ) -> np.ndarray:
        """
        Translates indices between the cleaned and original data representations.

        Args:
            indices (np.ndarray): The indices to translate.
            direction (Direction): The direction of translation.
        Returns:
            np.ndarray: The translated indices.
        """
        logger.debug("Translating indices.")
        if self._indices_translation_map is None:  # no translation needed
            return indices

        if direction == Direction.NEW_OLD:
            return self._indices_translation_map[indices, 0]
        return self._indices_translation_map[indices, 1]

    def _perform_enn(
        self, indices: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs Edited Nearest Neighbor (ENN) cleaning to remove potential outliers.

        Args:
            indices (np.ndarray): Indices of the training data.
            X (np.ndarray): Feature data.
            y (np.ndarray): Class labels.
            k (int, optional): Number of neighbors to consider for ENN. Defaults to 5.
            min_allowed (int, optional): Minimum number of matching neighbors required. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The cleaned features and corresponding labels.
        """
        logger.debug("Performing ENN cleaning.")

        indices = self._query_knn(
            indices, k=self.enn_neighbors, return_distances_=False
        )

        neighbor_labels = np.array(y[indices.ravel()]).reshape(indices.shape)
        keep_mask = (
            np.sum(neighbor_labels == np.array(y)[:, np.newaxis], axis=1)
            >= self.enn_min_matching_neighbors
        )
        X_clean = X[keep_mask]
        y_clean = y[keep_mask]
        logger.debug(f"Removed {len(X) - len(X_clean)} entries.")

        # create translation map
        # -1 means no translation (removed, thus not in the map)
        new_old_map = -np.ones(X.shape[0], dtype=int)
        old_new_map = -np.ones(X.shape[0], dtype=int)
        new_old_map[: X_clean.shape[0]] = np.flatnonzero(keep_mask)
        old_new_map[keep_mask] = np.arange(X_clean.shape[0])

        self._indices_translation_map = np.column_stack((new_old_map, old_new_map))
        self._removed_mask = ~keep_mask

        return X_clean, y_clean

    def _batch_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_history_: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        Predicts using the last batch of models in the ensemble.
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True labels.
            save_history_ (bool): Whether to save the prediction history.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the predictions and cached class predictions
        """
        if len(self.estimators_) < self.step_size:
            raise ValueError(
                "Not enough models trained. Please train more models before predicting."
            )

        preds, cached_class_preds = self._evaluate_batch(
            X,
            y,
            self.estimators_[-self.step_size :],  # last batch models
            cached_class_preds=cached_class_preds,
        )
        if save_history_:
            self._save_history_entry(
                f"Batch__{i}__preds",
                preds=preds,
                cached_class_preds=cached_class_preds,
            )

        return preds, cached_class_preds

    def _oversample(
        self,
        indices: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        minority_mask: np.ndarray,
        minority_mask_val: np.ndarray,
        class_preds: np.ndarray,
        border_mask: np.ndarray,
        step: int,
        save_history_: bool = False,
    ):
        with TimedLogger(
            f"Performing adaptive optimizations (step {step})",
            logger=logger,
            level=logging.INFO,
        ):
            r = self._identify_bp(
                X_val,
                y_val,
                minority_mask=minority_mask,
                minority_mask_val=minority_mask_val,
                class_preds=class_preds,
                return_history_=save_history_,
                **self._bp_kwargs,
            )
            if save_history_:
                bp_mask, bp_history = r
            else:
                bp_mask = r
                bp_history = {}

            # calculate regions to oversample
            if self.mask_merging_strategy == MergingStrategy.AND:
                oversampling_indices = indices[border_mask & bp_mask]
            else:
                oversampling_indices = indices[border_mask | bp_mask]

            X_to_oversample = X[oversampling_indices]
            if len(X_to_oversample) > 0:
                X_synth, y_synth = self._perform_smote(X, y, X_to_oversample)
            else:
                X_synth, y_synth = None, None
                logger.warning("No regions to oversample found.")

        if save_history_:
            self._save_history_entry(
                f"Batch__{step}__oversampling",
                oversampling_indices=oversampling_indices,
                X_synth=X_synth,
                y_synth=y_synth,
                **bp_history,
            )
        return X_synth, y_synth

    def _evaluate_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        executor: concurrent.futures.ThreadPoolExecutor,
        batch_models: List[ClassifierMixin],
        cached_class_preds: np.ndarray | None = None,
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
        logger.debug("Evaluating batch models (%d).", len(batch_models))

        batch_class_preds = np.array(
            list(executor.map(lambda clf: clf.predict(X), batch_models))
        )
        if cached_class_preds is not None:
            all_class_preds = np.concatenate(
                [cached_class_preds, batch_class_preds], axis=0
            )
        else:
            all_class_preds = batch_class_preds

        majority_vote = self.vote(
            all_class_preds,
            proba_=False,
        )
        return majority_vote, all_class_preds

    def _identify_bp(
        self,
        X: np.ndarray,
        y_val: np.ndarray,
        class_preds: np.ndarray,
        minority_mask: np.ndarray,
        minority_mask_val: np.ndarray,
        theta: float = 0.7,
        return_history_: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Identifies regions of poor performance (Bad Performance - BP) in the validation data.

        Warning:
            Implementation supports just binary classification.

        Args:
            X_clean (np.ndarray): Cleaned training samples.
            y_val (np.ndarray): True labels for the validation data.
            class_preds (np.ndarray): Predictions from the ensemble (for each model in enable,
                2D array of stacked predictions).
            minority_mask_val (np.ndarray): Mask for the minority class in the validation data.
            minority_class (int): The minority class label.
            theta (float): Confidence threshold.
            tau (float): Misclassification proportion threshold.
            return_history_ (bool): If True, returns the history of BP regions.

        Returns:
            np.ndarray: A boolean mask indicating the BP regions.
        """
        logger.debug("Identifying BP regions.")

        confidence_score = np.max(class_preds, axis=0)
        y_pred = np.argmax(class_preds, axis=0)

        # model is unsure of its prediction
        confidence_mask = confidence_score < theta
        # model is wrong
        miss_class_mask = y_pred != y_val

        val_bp_mask = confidence_mask | miss_class_mask
        val_bp_mask = val_bp_mask & minority_mask_val
        val_bp_indices = np.flatnonzero(val_bp_mask)

        # choose corresponding X samples
        minority_train_bp_indices = self._translate_val_to_train(
            val_bp_indices, return_distances_=False
        ).flatten()

        train_bp_mask = np.zeros(X.shape[0], dtype=bool)
        train_bp_mask[minority_mask][minority_train_bp_indices] = True

        if not return_history_:
            return train_bp_mask
        return train_bp_mask, {
            "confidence_mask": confidence_mask,
            "y_pred": y_pred,
            "miss_class_mask": miss_class_mask,
            "val_bp_mask": val_bp_mask,
            "bp_mask": train_bp_mask,
            "minority_train_bp_indices": minority_train_bp_indices,
            "minority_mask": minority_mask,
        }

    def _identify_br(
        self,
        indices: np.ndarray,
        y: np.ndarray,
        minority_mask: np.ndarray,
        br_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Identifies borderline regions (BR) for oversampling based on neighbor inconsistencies.

        Args:
            indices (np.ndarray): Indices of samples to query.
            y (np.ndarray): Class labels.
            minority_mask (np.ndarray): Mask for the minority class.
            br_threshold (float, optional): Threshold for determining borderline regions
                (e.g. 0.5 means at least 50% of neighbors are of a different class). Defaults to 0.5.

        Returns:
            np.ndarray: A boolean mask indicating borderline regions.
        """
        logger.debug("Identifying BR regions.")

        neighbors_indices = self._query_knn(
            indices, k=self.oversampling_neighbors, return_distances_=False
        )
        neighbor_labels = np.array(y[neighbors_indices.ravel()]).reshape(
            neighbors_indices.shape
        )

        br_mask = (
            np.sum(neighbor_labels == y[:, np.newaxis], axis=1)
            <= br_threshold * neighbor_labels.shape[1]
        ) & minority_mask
        return br_mask

    def _perform_smote(
        self, X: np.ndarray, y_clean: np.ndarray, X_to_oversample: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies SMOTE to generate synthetic samples for the minority class.

        Args:
            X (np.ndarray): Cleaned feature data.
            y_clean (np.ndarray): Cleaned labels.
            X_to_oversample (np.ndarray): Samples selected for oversampling.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The synthetic features and labels generated by SMOTE.
        """
        return self._smote._fit_resample(X, y_clean, X_to_oversample)
