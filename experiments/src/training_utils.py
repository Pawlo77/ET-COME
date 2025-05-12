"""Utility functions for training."""

import logging
import math
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce
from typing import Any, Callable, Dict, Iterator, List, Tuple

import kagglehub
import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from kagglehub import KaggleDatasetAdapter
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from tqdm.notebook import tqdm
from ucimlrepo import fetch_ucirepo

from .utils import DATA_DIR

RANDOM_SEED: int = 42

warnings.simplefilter("ignore")
np.random.seed(RANDOM_SEED)
logger = logging.getLogger(__name__)


# pylint: disable=all
class MyBaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom BaggingClassifier to handle the case when the base estimator is not fitted.
    """

    def __init__(
        self,
        base_estimator,
        n_estimators=10,
        max_samples=1.0,
        bootstrap=True,
        random_state=None,
        oversampler_class=None,
        threads=4,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.oversampler_class = oversampler_class
        self.threads = threads

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self.random_state_ = np.random.RandomState(self.random_state)

        def fit_estimator(i):
            # Bootstrap sample
            indices = self.random_state_.choice(
                np.arange(len(X)),
                size=int(self.max_samples * len(X)),
                replace=self.bootstrap,
            )
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

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)

        predictions = np.array([est.predict(X) for est in self.estimators_])
        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )
        return majority_votes

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X)

        probas = np.mean([est.predict_proba(X) for est in self.estimators_], axis=0)
        return probas


class FetchException(Exception):
    """Custom exception for dataset fetching errors."""


class BinaryDatasetManager:
    """Class to manage binary datasets for imbalanced learning experiments."""

    additional_datasets = {
        "Polish companies bankruptcy": {
            "source_func": partial(
                fetch_ucirepo,
                id=365,
            ),
            "DESCR": "Polish companies bankruptcy dataset",
        },
        "Breast Cancer Wisconsin (Diagnostic)": {
            "source_func": partial(
                fetch_ucirepo,
                id=17,
            ),
            "DESCR": "Breast Cancer Wisconsin (Diagnostic) dataset",
        },
        "Ionosphere": {
            "source_func": partial(
                fetch_ucirepo,
                id=52,
            ),
            "DESCR": "Ionosphere dataset",
        },
        "Pima Indians Diabetes": {
            "source_func": partial(
                kagglehub.load_dataset,
                KaggleDatasetAdapter.PANDAS,
                "uciml/pima-indians-diabetes-database",
                "diabetes.csv",
            ),
            "DESCR": "Pima Indians Diabetes dataset",
            "target": "Outcome",
        },
    }

    def __init__(self, data_home: str = DATA_DIR, random_state: int = RANDOM_SEED):
        """
        Initialize the BinaryDatasetManager with the specified data home and random state.

        Args:
            data_home (str): Directory to store datasets.
            random_state (int): Random seed for reproducibility.
        """
        logger.info("Fetching datasets from imbalanced-learn...")
        self.datasets = fetch_datasets(
            data_home=data_home,
            random_state=random_state,
            shuffle=True,
            verbose=True,
        )

        logger.info("Fetching additional datasets...")
        self.datasets.update(self._fetch_additional_datasets())

        logger.info("Preparing datasets...")
        self._prepare_datasets()

        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """
        Convert the datasets to a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing dataset information.
        """
        if self._df is None:
            self._df = pd.DataFrame(
                self.datasets.values(), index=self.datasets.keys()
            ).sort_values(by="inbalance strength", ascending=True)
        return self._df

    # pylint: disable=invalid-name,too-many-arguments
    # pylint: disable=too-many-locals,too-many-positional-arguments
    def serve(
        self,
        test_size: float = 0.2,
        test_to_valid_ratio: float = 0.5,
        random_state: int = RANDOM_SEED,
        oversampler: Any | None = None,
        preprocessing_pipeline_creator: (
            Callable[[np.ndarray], ColumnTransformer] | None
        ) = None,
    ) -> Iterator[
        Tuple[
            str,
            Tuple[
                Tuple[np.ndarray, np.ndarray],
                Tuple[np.ndarray, np.ndarray],
                Tuple[np.ndarray, np.ndarray],
            ],
        ]
    ]:
        """
        Serve the datasets for training and testing.
        This method splits the datasets into training and testing sets,
        performs preprocessing, and applies oversampling if specified.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            test_to_valid_ratio (float): Ratio of test to validation set size.
            random_state (int): Random seed for reproducibility.
            oversampler (Any | None): Oversampling method to apply to the training set.
            preprocessing_pipeline_creator (Callable[[np.ndarray], ColumnTransformer] | None):
                Function to create a preprocessing pipeline that takes a feature matrix and returns
                a ColumnTransformer.
        Yields:
            Tuple[str, Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
                Dataset name and a tuple containing training, validating and testing sets.
        Raises:
            ValueError: If the dataset is not binary.
        """
        if preprocessing_pipeline_creator is None:
            preprocessing_pipeline_creator = self.get_preprocessing_pipeline

        for dataset_name, (X, y) in self._df[["data", "target"]].iterrows():
            logger.debug("Serving dataset: %s", dataset_name)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            X_test, X_valid, y_test, y_valid = train_test_split(
                X_test,
                y_test,
                test_size=test_to_valid_ratio,
                stratify=y_test,
                random_state=random_state,
            )

            logger.debug(
                "Train shape: %s, Valid shape: %s, Test shape: %s",
                str(X_train.shape),
                str(X_valid.shape),
                str(X_test.shape),
            )

            pipeline = preprocessing_pipeline_creator(X_train)
            X_train = pipeline.fit_transform(X_train)
            X_valid = pipeline.transform(X_valid)
            X_test = pipeline.transform(X_test)

            if oversampler is not None:
                X_train, y_train = oversampler.fit_resample(X_train, y_train)
                logger.debug(
                    "Applied oversampling: Train shape: %s, Test shape: %s",
                    str(X_train.shape),
                    str(X_test.shape),
                )

            yield (
                dataset_name,
                (
                    (X_train, y_train),
                    (X_valid, y_valid),
                    (X_test, y_test),
                ),
            )

    def __len__(self) -> int:
        """
        Get the number of datasets.

        Returns:
            int: Number of datasets.
        """
        return len(self.datasets)

    def _prepare_datasets(self) -> None:
        """
        Analyze the datasets to check for class imbalance.
        """
        for val in self.datasets.values():
            val["target"] = np.where(val["target"] == 1, 1, 0)
            unique_vals, counts = np.unique(val["target"], return_counts=True)
            val["counts"] = dict(zip(unique_vals, counts))

            val["class negative num"] = class_negative_num = val["counts"][0]
            val["class positive num"] = class_positive_num = val["counts"][1]
            gcd_all = reduce(math.gcd, [class_negative_num, class_positive_num])
            val["ratio (0 : 1)"] = (
                f"{int(class_negative_num / gcd_all)}:{int(class_positive_num / gcd_all)}"
            )
            val["shape"] = val["data"].shape
            val["inbalance strength"] = round(
                max(class_negative_num, class_positive_num)
                / min(class_negative_num, class_positive_num),
                2,
            )

        return self

    # pylint: disable=magic-value-comparison
    def _fetch_additional_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch additional datasets from UCI ML repository and Kaggle.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary containing additional datasets.
        Raises:
            Exception: If fetching the dataset fails.
        """
        additional_datasets = {}
        for dataset_name, dataset_info in self.additional_datasets.items():
            try:
                dataset = dataset_info["source_func"]()

                if "target" in dataset_info:
                    target = dataset[dataset_info["target"]]
                    data = dataset.drop(columns=[dataset_info["target"]])
                else:
                    target = dataset.data.targets
                    data = dataset.data.features

                target = np.array(target)
                data = np.array(data)

                # convert to proper format
                unique_values = sorted(np.unique(target))
                if len(unique_values) != 2:
                    raise ValueError(
                        (
                            "Target variable must have exactly 2 "
                            f"unique values, found {len(unique_values)}"
                        )
                    )
                target[target == unique_values[0]] = -1
                target[target == unique_values[1]] = 1

                additional_datasets[dataset_name] = {
                    "DESCR": dataset_info["DESCR"],
                    "target": target,
                    "data": data,
                }
            except Exception as e:
                raise FetchException(f"Failed to fetch {dataset_name}: {e}") from e

        return additional_datasets

    @staticmethod
    def get_preprocessing_pipeline(X: np.array) -> ColumnTransformer:
        """
        Create a preprocessing pipeline for the dataset.
        This includes imputation and scaling for numeric features,
        and one-hot encoding for categorical features.

        Args:
            X (np.array): Feature matrix.
        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """

        numeric_features = [
            i for i in range(X.shape[1]) if np.issubdtype(X[:, i].dtype, np.number)
        ]
        categorical_features = [
            i for i in range(X.shape[1]) if not np.issubdtype(X[:, i].dtype, np.number)
        ]

        # Define transformers
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combine preprocessing steps
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )


# pylint: disable=invalid-name
def evaluate_wrapper(
    args: Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    option: str,
    base_estimator_class: Callable[..., BaseEstimator],
    scorers: Dict[str, Any],
    random_state: int = RANDOM_SEED,
    bagging_classifier_class: Callable[..., BaseEstimator] | None = None,
) -> Dict[str, Any]:
    """
    Evaluate a single parameter setting using the ParamRunner.evaluate method.

    Args:
        args:
            A tuple containing:
                - params (Dict[str, Any]): Parameters for the model.
                - X_train (np.ndarray): Training features.
                - y_train (np.ndarray): Training targets.
                - X_test (np.ndarray): Testing features.
                - y_test (np.ndarray): Testing targets.
        option (str): Evaluation option (e.g., "bagging").
        base_estimator_class (Callable[..., BaseEstimator]):
            Class of the base estimator to instantiate.
        scorers (Dict[str, Any]): Dictionary of scorer functions.
        random_state (int): Random seed for reproducibility.
        bagging_classifier_class (Callable[..., BaseEstimator] | None):
            Class of the bagging classifier to instantiate.
    Returns:
        Dict[str, Any]: Dictionary of evaluation results.
    """
    params, X_train, y_train, X_test, y_test = args
    return ParamRunner.evaluate(
        params,
        X_train,
        y_train,
        X_test,
        y_test,
        option,
        base_estimator_class,
        scorers,
        random_state,
        bagging_classifier_class=bagging_classifier_class,
    )


# pylint: disable=too-many-instance-attributes
class ParamRunner(BaseEstimator):
    """
    A parameter runner class for model evaluation over a grid of parameters.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        base_estimator_class: Callable[..., BaseEstimator],
        param_grid: Dict[str, Any],
        scoring: Dict[str, str],
        option: str,
        random_state: int = RANDOM_SEED,
        bagging_classifier_class: Callable[..., BaseEstimator] | None = None,
        n_jobs: int = 1,
    ) -> None:
        """
        Initialize the ParamRunner.

        Args:
            base_estimator_class (Callable[..., BaseEstimator]): Class of the base estimator.
            param_grid (Dict[str, Any]): Dictionary representing the parameter grid.
            scoring (Dict[str, str]): Dictionary of scoring metrics.
            option (str): Option for evaluation (e.g., "bagging").
            random_state (int, optional): Random seed for reproducibility.
            n_jobs (int, optional): Number of jobs for parallel processing. Defaults to 1.
            bagging_classifier_class (Callable[..., BaseEstimator] | None): Class of the
                bagging classifier to instantiate.
        """
        self.base_estimator_class = base_estimator_class
        self.bagging_classifier_class = bagging_classifier_class

        self.param_grid = param_grid
        self.scoring = scoring

        self.option = option

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.results_: pd.DataFrame | None = None
        self._scorers: Dict[str, Any] | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> "ParamRunner":
        """
        Evaluate all parameter combinations on training and testing data.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing targets.
        Returns:
            ParamRunner: The fitted ParamRunner instance with results saved in results_.
        """
        self._scorers: Dict[str, Any] = {
            name: get_scorer(metric) for name, metric in self.scoring.items()
        }
        param_list: List[Dict[str, Any]] = list(ParameterGrid(self.param_grid))

        if self.n_jobs > 1 or self.n_jobs == -1:
            tasks: List[
                Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ] = [(params, X_train, y_train, X_test, y_test) for params in param_list]
            results = self._run_parallel_evaluation(
                tasks,
                self.n_jobs,
                len(tasks),
                self.option,
                self.base_estimator_class,
                self._scorers,
                self.random_state,
                bagging_classifier_class=self.bagging_classifier_class,
            )
        else:
            results = []
            for params in tqdm(param_list, desc="Parameter Grid"):
                result = self.evaluate(
                    params,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    self.option,
                    self.base_estimator_class,
                    self._scorers,
                    self.random_state,
                    bagging_classifier_class=self.bagging_classifier_class,
                )
                results.append(result)

        self.results_ = pd.DataFrame(results)
        return self

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,magic-value-comparison
    @staticmethod
    def evaluate(
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        option: str,
        base_estimator_class: Callable[..., BaseEstimator],
        scorers: Dict[str, Any],
        random_state: int = RANDOM_SEED,
        bagging_classifier_class: Callable[..., BaseEstimator] | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single set of parameters with training and testing data.

        Args:
            params (Dict[str, Any]): Parameter set.
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing targets.
            option (str): Evaluation option (e.g., "bagging").
            base_estimator_class (Callable[..., BaseEstimator]):
                Class of the base estimator.
            scorers (Dict[str, Any]): Dictionary of scorer functions.
            random_state (int): Random seed for reproducibility.
            bagging_classifier_class (Callable[..., BaseEstimator] | None):
                Class of the bagging classifier to instantiate.
        Returns:
            Dict[str, Any]: Dictionary containing parameter configuration and scores.
        Raises:
            ValueError: If n_estimators is not in params for bagging.
        """
        # Instantiate model with or without bagging.
        if option == "bagging":
            if "n_estimators" not in params:
                raise ValueError("n_estimators must be in params for bagging.")
            if bagging_classifier_class is None:
                raise ValueError(
                    "bagging_classifier_class must be provided for bagging."
                )
            n_estimators = params.pop("n_estimators")
            model = bagging_classifier_class(
                base_estimator=base_estimator_class(
                    **params, random_state=random_state
                ),
                n_estimators=n_estimators,
                random_state=random_state,
            )
        else:
            model = base_estimator_class(**params, random_state=random_state)
        model.fit(X_train, y_train)

        scores: Dict[str, Any] = {"params": params}
        for mode, X, y in zip(["train", "test"], [X_train, X_test], [y_train, y_test]):
            mode_scores = {}
            for name, scorer in scorers.items():
                if name == "roc_auc":
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X)[:, 1]
                        mode_scores[name] = scorer._score_func(y, y_proba)  # pylint: disable=protected-access
                    else:
                        mode_scores[name] = None
                else:
                    mode_scores[name] = scorer(model, X, y)
            scores[mode] = mode_scores

        return scores

    @staticmethod
    def _run_parallel_evaluation(
        tasks: List[
            Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
        n_jobs: int,
        total: int,
        option: str,
        base_estimator_class: Callable[..., BaseEstimator],
        scorers: Dict[str, Any],
        random_state: int = RANDOM_SEED,
        bagging_classifier_class: Callable[..., BaseEstimator] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Run evaluations in parallel using multiprocessing.

        Args:
            tasks (List[Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
                A list of tasks where each task is a tuple with parameters and dataset splits.
            n_jobs (int): Number of processes to run in parallel.
            total (int): Total number of tasks.
            option (str): Evaluation option (e.g., "bagging").
            base_estimator_class (Callable[..., BaseEstimator]):
                Class of the base estimator.
            scorers (Dict[str, Any]): Dictionary of scorer functions.
            random_state (int): Random seed for reproducibility.
            bagging_classifier_class (Callable[..., BaseEstimator] | None):
                Class of the bagging classifier to instantiate.
        Returns:
            List[Dict[str, Any]]: List of evaluation results.
        """
        processes = n_jobs if n_jobs != -1 else os.cpu_count()
        with mp.Pool(processes=processes) as pool:
            func = partial(
                evaluate_wrapper,
                option=option,
                base_estimator_class=base_estimator_class,
                bagging_classifier_class=bagging_classifier_class,
                scorers=scorers,
                random_state=random_state,
            )
            results: List[Dict[str, Any]] = []
            for result in tqdm(
                pool.imap_unordered(func, tasks), total=total, desc="Parameter Grid"
            ):
                results.append(result)
        return results
