# pylint: disable=c-extension-no-member
"""
A custom ensemble classifier for imbalanced data.
It leverages approximate KNN using HNSWlib for fast neighbor searches,
applies Edited Nearest Neighbor (ENN) cleaning
and uses iterative training with oversampling.
The classifier is compatible with scikit-learn and can use any base estimator.
"""

import concurrent.futures
import logging
import math
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import hnswlib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y
from tqdm import tqdm

from bagging_classifier import BaggingClassifier
from efficient_smote import EfficientSMOTE
from utils import TimedLogger, create_logger

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


# pylint: disable=too-many-instance-attributes,invalid-name
class InspireClassifier(BaggingClassifier):
    """
    A custom ensemble classifier using iterative
    training with ENN cleaning and oversampling via SMOTE-BORDERLINE.
    Base classifiers are trained on the cleaned data
    in bagging terminology.
    """

    # pylint: disable=dangerous-default-value,too-many-arguments
    # pylint: disable=too-many-positional-arguments,too-many-locals
    # pylint: disable=too-many-statements
    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        bootstrap: bool = True,
        random_state: int = None,
        threads: int = 4,
        minority_class: Any | None = None,
        #
        enn_neighbors: int = 7,
        enn_min_matching_neighbors: int = 3,
        oversampling_neighbors: int = 10,
        val_to_train_neighbors: int = 3,
        knn_kw: dict = {},
        approximate_knn_: bool = True,
        duplicate_out_of_cache_entries_: bool = True,
        #
        step_size: int = 2,
        cache_size: int | None = None,
        #
        oversampling_per_step: int | None = None,
        oversampling_ratio: float | None = 0.3,
        adaptive_oversampling_step: bool = False,
        #
        br_threshold: float = 0.5,
        bp_theta: float = 0.7,
        mask_merging_strategy: MergingStrategy = MergingStrategy.AND,
        #
        verbose_: bool = False,
        remove_outliers_: bool = True,
        logging_level: int = logging.INFO,
    ) -> None:
        """
        Initializes the custom ensemble classifier.

        Args:
            base_estimator (BaseEstimator): The base estimator to fit on
                random subsets of the data.
            n_estimators (int): The number of base estimators in the ensemble.
            max_samples (float): The fraction of samples to draw from X to
                train each base estimator.
            bootstrap (bool): Whether samples are drawn with replacement.
            random_state (int): Random seed for reproducibility.
            threads (int): Number of threads to use for parallel processing.

            enn_neighbors (int): Number of neighbors for
                Edited Nearest Neighbor (ENN) cleaning.
            enn_min_matching_neighbors (int): Minimum number of matching
                neighbors for ENN cleaning.
            oversampling_neighbors (int): Number of neighbors for oversampling.
            val_to_train_neighbors (int): Number of neighbors for validation
                to training set mapping.
            knn_kw (dict): Keyword arguments for KNN.
            approximate_knn_ (bool): Whether to use approximate KNN.
            duplicate_out_of_cache_entries_ (bool): Whether to duplicate
                proper entries in cache if cache is not large enough.
                If False, raises an error if cache is not large enough.

            step_size (int): Number of models to train in each step.
            cache_size (int | None): Size of the KNN cache. If not provided,
                it will be set to the maximum of enn_neighbors and
                oversampling_neighbors.

            oversampling_per_step (int | None): Number of samples to oversample in each step.
            oversampling_ratio (float | None): Ratio of samples to oversample
                based on the minority class. If provided, it will oversample
                ratio * n_minority samples (valid, ie after ENN cleaning).
            adaptive_oversampling_step (bool): Whether to adaptively adjust
                the oversampling step size. If enabled, number of oversampling
                samples will be adjusted as number of True entries in the
                oversampling masks (joined bp_mask and br_mask) times
                oversampling_ratio.

            br_threshold (float): Threshold for identifying border regions.
                (e.g. 0.5 means at least 50% of neighbors are of a different class).
            bp_theta (float): Confidence threshold for identifying bad performance regions.

            mask_merging_strategy (MergingStrategy): Strategy for merging masks.
            verbose_ (bool): Whether to print progress.
            approximate_knn_ (bool): Whether to use approximate KNN.
            remove_outliers_ (bool): Whether to remove outliers using ENN cleaning.
        Raises:
            ValueError: If any of the parameters are invalid.
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
        if not (oversampling_per_step or oversampling_ratio):
            raise ValueError(
                "Specify only one of 'oversampling_per_step' or 'oversampling_ratio'."
            )
        if "k" in knn_kw:  # pylint: disable=magic-value-comparison
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
                "cache_size must be greater than or equal to "
                "max(enn_neighbors, oversampling_neighbors)."
            )
        if enn_min_matching_neighbors > enn_neighbors:
            raise ValueError(
                "enn_min_matching_neighbors must be less than or equal to enn_neighbors."
            )
        if adaptive_oversampling_step and not oversampling_ratio:
            raise ValueError(
                "adaptive_oversampling_step requires oversampling_ratio to be set."
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
        self.adaptive_oversampling_step = adaptive_oversampling_step
        self.duplicate_out_of_cache_entries_ = duplicate_out_of_cache_entries_

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

        self.logging_level = logging_level
        logger.setLevel(logging_level)

    @property
    def history(self) -> List[dict]:
        """Returns the training history, if available."""
        return self._history

    @history.deleter
    def history(self):
        """Deletes the training history."""
        del self._history

    # pylint: disable=arguments-differ
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
            remove_cache_ (bool): Whether to clean up cached data
                after training (excluding training history).
            save_history_ (bool): Whether to save results from
                each training step (can be space/time intensive).
        Returns:
            InspireClassifier: The fitted classifier instance.
        Raises:
            RuntimeError: If the classifier is already fitted.
            ValueError: If n_estimators is not divisible by
                step_size or if bp_kwargs does not contain 'neighbors'.
        """
        if self._fitted_:
            raise ValueError("This classifier instance is already fitted.")

        X, y = check_X_y(X, y)
        X_val, y_val = check_X_y(X_val, y_val)

        # implementation is expecting 1D target arrays
        if len(y.shape) == 2:  # pylint: disable=magic-value-comparison
            y = y.ravel()
        if len(y_val.shape) == 2:  # # pylint: disable=magic-value-comparison
            y_val = y_val.ravel()

        if save_history_:
            self._history = []

        logger.debug("Starting fit process.")

        # Step 1: Fit and cache the KNN index for the training set
        # (between training set itself)
        self._fit_knn(X=X)

        # Step 2: Perform ENN cleaning on the training data
        if self.remove_outliers_:
            X, y = self._perform_enn(
                np.arange(len(X)), X, y, save_history_=save_history_
            )
        indices = np.arange(len(X))  # new indices after ENN

        # Step 3: identify target classes, their ratios and identify the minority class
        # (if not provided)
        self.classes_, counts = np.unique(y, return_counts=True)
        ratios = self.get_class_ratio(y)
        if self.minority_class is None:
            self.minority_class = self.classes_[np.argmin(counts)]
            logger.debug("Identified minority class: %s", self.minority_class)

        # identify masks for the minority class
        minority_mask = y == self.minority_class
        minority_mask_val = y_val == self.minority_class

        # Step 5: Identify border regions for oversampling.
        br_mask = self._identify_br(
            indices=indices,
            y=y,
            minority_mask=minority_mask,
            save_history_=save_history_,
        )

        # Step x: Clear KNN cache.
        # Fit and cache the KNN index for minority class in training set.
        del self._full_knn_indices, self._full_knn_distances
        self._knn_fitted_ = False
        self._indices_translation_map = None
        
        X_minority = X[minority_mask]
        self._fit_knn(X=X_minority, **self.knn_kw)
        
        # Step 4: Fit and cache the KNN index for the validation set
        # (between validation set and minority class in cleaned training set
        # as no more is needed)
        self._fit_knn_val(X=X_minority, X_val=X_val, **self.knn_kw)

        # Determine oversampling_per_step based on the oversampling ratio
        # if adaptive_oversampling_step is not enabled
        if not self.adaptive_oversampling_step and (
            not self.oversampling_per_step and self.oversampling_ratio
        ):
            n_minority = sum(minority_mask)
            n_majority = len(minority_mask) - n_minority
            oversampling_per_step = int(n_majority * self.oversampling_ratio - n_minority)
            
            if oversampling_per_step < 0:
                raise ValueError("The specified oversampling_ratio is too low: it results in fewer minority samples than majority samples after oversampling. Please increase the oversampling_ratio or check your class distribution.")

        # initialize smote
        self._smote = EfficientSMOTE(
            knn_callable=self._query_knn_cache,
            oversampling_per_step = oversampling_per_step,
            k_neighbors=self.oversampling_neighbors,
            minority_class=self.minority_class,
        )

        # training loop variables
        all_models_cached_preds = None
        X_synth, y_synth = None, None

        # Step 5: Iterative training using parallel execution.
        gen = range(math.ceil(self.n_estimators / self.step_size))
        if self.verbose_:
            gen = tqdm(
                gen,
                desc="Training progress",
            )
        with concurrent.futures.ThreadPoolExecutor(self.threads) as executor:
            for step in gen:
                if step > 0:
                    # Step 5.1: Calculate new models predictions on the validation set
                    # and add them to the cache.
                    y_proba, all_models_cached_preds = self._batch_predict(
                        idx=step,
                        X=X_val,
                        executor=executor,
                        all_models_cached_preds=all_models_cached_preds,
                        save_history_=save_history_,
                    )

                    # 5.2: Perform oversampling
                    X_synth, y_synth = self._perform_oversampling(
                        y_proba=y_proba,
                        indices=indices,
                        X=X_minority,
                        y=y[minority_mask],
                        y_val=y_val,
                        minority_mask=minority_mask,
                        minority_mask_val=minority_mask_val,
                        br_mask=br_mask,
                        step=step,
                        save_history_=save_history_,
                    )

                # Step 5.3: Train next models batch
                models_to_train = (
                    self.step_size
                    if step < len(gen) - 1
                    else (self.n_estimators - step * self.step_size)
                )
                with TimedLogger(
                    f"Training models (step {step})",
                    logger=logger,
                    level=logging.INFO,
                ):
                    futures = [
                        executor.submit(
                            self._train_model,
                            idx=i,
                            X=X,
                            y=y,
                            ratios=ratios,
                            X_synth=X_synth,
                            y_synth=y_synth,
                        )
                        for i in range(models_to_train)
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

        with TimedLogger("Fitting KNN index", logger=logger, level=logging.INFO):
            # between X and X
            indices, distances = self._fit_knn_helper(
                X=X, X_query=X, k=self.cache_size + 1
            )
            # Do not store the first index as it is the same observation.
            self._full_knn_indices, self._full_knn_distances = (
                indices[:, 1:],
                distances[:, 1:],
            )

        self._knn_fitted_ = True

    def _fit_knn_val(self, X: np.ndarray, X_val: np.ndarray, **knn_kw) -> None:
        """
        Fits a KNN index for the validation set and caches the results.
        Between X_val and X, used in `_translate_val_to_train`.

        Args:
            X (np.ndarray): Training data.
            X_val (np.ndarray): Validation data.
            **knn_kw: Keyword arguments for KNN fitting.
        Raises:
            RuntimeError: If the validation KNN index is already fitted.
        """
        if self._knn_val_fitted_:
            raise RuntimeError("KNN index for validation set already fitted.")

        with TimedLogger(
            "Fitting KNN index for validation set",
            logger=logger,
            level=logging.INFO,
        ):
            self._knn_val_indices, self._knn_val_distances = self._fit_knn_helper(
                X=X, X_query=X_val, k=self.val_to_train_neighbors
            )

        self._knn_val_fitted_ = True

    def _fit_knn_helper(
        self,
        X: np.ndarray,
        X_query: np.ndarray,
        k: int = 5,
        ef: int = 500,
        ef_construction: int = 200,
        M: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits a KNN index using either approximate or exact
        search and returns the indices and distances.

        Args:
            X (np.ndarray): Training data to index.
            X_query (np.ndarray): Data to query the index.
            approximate_knn_ (bool): Whether to use approximate KNN.
            k (int, optional): Number of neighbors to retrieve.
            ef (int, optional): HNSWlib ef parameter.
            ef_construction (int, optional): HNSWlib construction parameter.
            M (int, optional): HNSWlib M parameter.
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

    def _query_knn_cache(
        self, indices: np.ndarray, return_distances_: bool = True
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Extracts the KNN indices and distances from the cache for the training set.
        Indices are expected to be in new (ie. after ENN) representation.

        Args:
            indices (np.ndarray): Indices in to query.
            return_distances_ (bool, optional): If True, also returns
                neighbor distances.
        Returns:
            Tuple[np.ndarray, np.ndarray] | np.ndarray: A tuple
                (neighbor_indices, neighbor_distances) if return_distances_ is True,
                or just the neighbor indices.
        Raises:
            RuntimeError: If the training KNN index is not yet fitted or
                if the cache is not large enough.
        """
        if not self._knn_fitted_:
            raise RuntimeError("KNN index not fitted. Call _fit_knn first.")
        logger.debug("Using cached training KNN results.")

        indices = self._translate_indices(indices, direction=Direction.NEW_OLD)
        neighbors_indices = self._full_knn_indices[indices]

        # filter out removed samples
        if self._removed_mask is not None:
            mask = self._removed_mask[neighbors_indices]
            filtered = []
            for row, row_mask in zip(neighbors_indices, mask):
                valid: np.ndarray = row[~row_mask]
                if valid.size < self.oversampling_neighbors:
                    if self.duplicate_out_of_cache_entries_:
                        logger.warning(
                            "Less than k valid neighbors available in at least one row. "
                            "Duplicating valid neighbors to fill the gap."
                        )
                        valid = np.concatenate(
                            [valid, np.tile(valid, self.oversampling_neighbors)]
                        )
                    else:
                        raise RuntimeError(
                            "Less than k valid neighbors available in at least one row."
                        )  # this means that the cache is not large enough
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

    def _translate_val_to_train_minority(
        self, indices: np.ndarray, return_distances_: bool = True
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Queries the validation KNN index and returns the indices and distances
        of nearest observations in the training set.
        Translated indices are in the new representation (ie. after ENN)
        cropped only to the minority class.

        Args:
            indices (np.ndarray): Indices in the validation set to query.
            return_distances_ (bool, optional): If True, also returns neighbor
                distances.
        Returns:
            Tuple[np.ndarray, np.ndarray] | np.ndarray: A tuple
                (neighbor_indices, neighbor_distances) if return_distances_ is True,
                or just the neighbor indices.
        Raises:
            RuntimeError: If the validation KNN index is not yet fitted.
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
        if self._indices_translation_map is None:  # no translation needed
            return indices

        logger.debug("Translating indices.")

        if direction == Direction.NEW_OLD:
            return self._indices_translation_map[indices, 0]
        return self._indices_translation_map[indices, 1]

    def _batch_predict(
        self,
        idx: int,
        X: np.ndarray,
        executor: concurrent.futures.ThreadPoolExecutor,
        all_models_cached_preds: np.ndarray | None = None,
        save_history_: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preforms batch prediction using models from the last step and
        saves them to the cache. Performs soft voting if save_history_ is True
        and saves the results to the history.

        Args:
            idx (int): Index of the current batch.
            X (np.ndarray): Validation features.
            executor (concurrent.futures.ThreadPoolExecutor): Executor for parallel processing.
            all_models_cached_preds (np.ndarray | None): Cached class predictions
                from previous models.
            save_history_ (bool): Whether to save the prediction history.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The predicted class probabilities
                and the cached class predictions.
        """
        if len(self.estimators_) < self.step_size:
            raise ValueError(
                "Not enough models trained. Please train more models before predicting."
            )
        batch_models = self.estimators_[-self.step_size :]

        logger.debug("Evaluating batch models (%d).", len(batch_models))

        # current batch predictions
        batch_models_preds = np.array(
            list(executor.map(lambda clf: clf.predict(X), batch_models))
        )

        # add to cache (for each row add the class prediction from each model)
        # - extend cache column-wise
        if all_models_cached_preds is not None:
            all_class_preds = np.concatenate(
                [all_models_cached_preds, batch_models_preds], axis=0
            )
        else:
            all_class_preds = batch_models_preds

        # soft voting
        y_proba = self.vote(
            all_class_preds,
            proba_=True,
        )

        if save_history_:
            self._save_history_entry(
                f"Batch__{idx}__preds",
                preds=y_proba,
                cached_class_preds=all_models_cached_preds,
            )
        return y_proba, all_class_preds

    def _identify_bp(
        self,
        train_size: int,
        y_proba: np.ndarray,
        y_val: np.ndarray,
        minority_mask: np.ndarray,
        minority_mask_val: np.ndarray,
        return_history_: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Identifies regions of poor performance (Bad Performance - BP) in the validation data.

        Warning:
            Implementation supports just binary classification.
        Args:
            train_size (int): Size of the training set.
            y_proba (np.ndarray): Class probabilities for the validation set.
            y_val (np.ndarray): True labels for the validation set.
            minority_mask (np.ndarray): Mask for the minority class in the training set.
            minority_mask_val (np.ndarray): Mask for the minority class in the validation set.
            return_history_ (bool, optional): If True, returns additional
                information about BP identification.
        Returns:
            np.ndarray: A boolean mask indicating the BP regions.
        """
        logger.debug("Identifying BP regions.")

        confidence_score = np.max(y_proba, axis=1)
        y_pred = np.argmax(y_proba, axis=1)

        # model is unsure of its prediction
        confidence_mask = confidence_score < self.bp_theta
        # model is wrong
        miss_class_mask = y_pred != y_val

        # combine masks - regions where model is unsure or wrong
        # filter out only minority class samples
        val_bp_mask = minority_mask_val & (confidence_mask | miss_class_mask)

        val_bp_indices = np.flatnonzero(val_bp_mask)

        # choose corresponding X samples
        minority_train_bp_indices = self._translate_val_to_train_minority(
            val_bp_indices, return_distances_=False
        ).flatten()
        
        minority_train_bp_mask = np.zeros(sum(minority_mask), dtype=bool)
        minority_train_bp_mask[minority_train_bp_indices] = True
            
        if not return_history_:
            return minority_train_bp_mask
        return minority_train_bp_mask, {
            "confidence_mask": confidence_mask,
            "y_pred": y_pred,
            "miss_class_mask": miss_class_mask,
            "val_bp_mask": val_bp_mask,
            "minority_train_bp_mask": minority_train_bp_mask,
            "minority_train_bp_indices": minority_train_bp_indices,
            "minority_mask": minority_mask,
        }

    def _identify_br(
        self,
        indices: np.ndarray,
        y: np.ndarray,
        minority_mask: np.ndarray,
        save_history_: bool = False,
    ) -> np.ndarray:
        """
        Identifies borderline regions (BR) for oversampling based on neighbor inconsistencies.
        Based on borderline SMOTE.

        Args:
            indices (np.ndarray): Indices of samples to query knn with.
            y (np.ndarray): Class labels.
            minority_mask (np.ndarray): Mask for the minority class.
            save_history_ (bool): Whether to save the BR identification history.
        Returns:
            np.ndarray: A boolean mask indicating borderline regions.
        """
        with TimedLogger(
            "Identifying border regions", logger=logger, level=logging.INFO
        ):
            neighbors_indices: np.ndarray = self._query_knn_cache(
                indices, return_distances_=False
            )
            neighbor_labels = np.array(y[neighbors_indices.ravel()]).reshape(
                neighbors_indices.shape
            )

            neighbor_minority_labels = neighbor_labels[minority_mask]
            # calculate number of neighbors in the same class as the original sample,
            # if lower than threshold * 100%, mark as borderline
            # filter out only minority class samples
            br_mask = np.zeros(sum(minority_mask), dtype=bool)

            br_mask[
                np.sum(neighbor_minority_labels == y[minority_mask, np.newaxis], axis=1)
                <= self.br_threshold * neighbor_minority_labels.shape[1]
            ] = True
            
        if save_history_:
            self._save_history_entry("BR", border_mask=br_mask)
        return br_mask

    def _perform_enn(
        self,
        indices: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        save_history_: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs Edited Nearest Neighbor (ENN) cleaning to remove potential outliers.

        Args:
            indices (np.ndarray): Indices of the training data.
            X (np.ndarray): Feature data.
            y (np.ndarray): Class labels.
            save_history_ (bool): Whether to save the ENN cleaning history.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The cleaned features and corresponding labels.
        """

        with TimedLogger("Removing outliers", logger=logger, level=logging.INFO):
            indices = self._query_knn_cache(indices, return_distances_=False)
            neighbor_labels = np.array(y[indices.ravel()]).reshape(indices.shape)

            # calculate number of neighbors in the same class as the original sample,
            # mark as outlier if less than min_allowed are found
            keep_mask = (
                np.sum(neighbor_labels == y[:, np.newaxis], axis=1)
                >= self.enn_min_matching_neighbors
            )

            # remove outliers
            X_clean = X[keep_mask]
            y_clean = y[keep_mask]
            logger.debug("Removed %d entries.", len(X) - len(X_clean))

            # create translation map
            # -1 means no translation (removed, thus not in the map),
            # this is fully redundant (those should not be accessed at all),
            # but for consistency
            new_old_map = -np.ones(X.shape[0], dtype=int)
            old_new_map = -np.ones(X.shape[0], dtype=int)
            new_old_map[: X_clean.shape[0]] = np.flatnonzero(keep_mask)
            old_new_map[keep_mask] = np.arange(X_clean.shape[0])
            self._indices_translation_map = np.column_stack((new_old_map, old_new_map))
            self._removed_mask = ~keep_mask

        if save_history_:
            self._save_history_entry("ENN", removed_mask=self._removed_mask)
        return X_clean, y_clean

    def _perform_smote(
        self, X: np.ndarray, y: np.ndarray, X_to_oversample: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies SMOTE to generate synthetic samples for the minority class.

        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Class labels.
            X_to_oversample (np.ndarray): Samples selected for oversampling.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The synthetic features and labels generated by SMOTE.
        """
        return self._smote._fit_resample(
            X, y, X_to_oversample
        )  # pylint: disable=protected-access

    def _perform_oversampling(
        self,
        indices: np.ndarray,
        y_proba: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        y_val: np.ndarray,
        minority_mask: np.ndarray,
        minority_mask_val: np.ndarray,
        br_mask: np.ndarray,
        step: int,
        save_history_: bool = False,
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """
        Performs oversampling on the identified regions.

        Args:
            indices (np.ndarray): Indices of the training data.
            y_proba (np.ndarray): Class probabilities from the ensemble.
            X (np.ndarray): Feature data.
            y (np.ndarray): Class labels.
            y_val (np.ndarray): Validation labels.
            minority_mask (np.ndarray): Mask for the minority class in the training set.
            minority_mask_val (np.ndarray): Mask for the minority class in the validation set.
            br_mask (np.ndarray): Mask for border regions.
            step (int): Current training step.
            save_history_ (bool): Whether to save the oversampling history.
        Returns:
            Tuple[np.ndarray | None, np.ndarray | None]: A tuple containing
                the synthetic features and labels.
        """
        with TimedLogger(
            f"Performing adaptive optimizations (step {step})",
            logger=logger,
            level=logging.INFO,
        ):
            # calculate bad performance regions
            r = self._identify_bp(
                y_proba=y_proba,
                train_size=len(X),
                y_val=y_val,
                minority_mask=minority_mask,
                minority_mask_val=minority_mask_val,
                return_history_=save_history_,
            )
            if save_history_:
                bp_mask, bp_history = r
            else:
                bp_mask = r
                bp_history = {}

            # calculate regions to oversample
            if self.mask_merging_strategy == MergingStrategy.AND:
                oversampling_mask = br_mask & bp_mask
            else:
                oversampling_mask = br_mask | bp_mask

            # perform oversampling on the identified regions
            indices_to_oversample = np.flatnonzero(oversampling_mask)
            
            if len(indices_to_oversample) > 0:
                X_synth, y_synth = self._perform_smote(X, y, indices_to_oversample)
            else:
                logger.info("No regions to oversample found.")
                X_synth, y_synth = None, None

        if save_history_:
            self._save_history_entry(
                f"Batch__{step}__oversampling",
                oversampling_mask=oversampling_mask,
                X_synth=X_synth,
                y_synth=y_synth,
                **bp_history,
            )
        return X_synth, y_synth
