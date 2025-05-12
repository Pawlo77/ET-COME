"""Dataset manager for binary datasets."""

import math
import warnings
from functools import partial, reduce
from typing import Any, Callable, Dict, Iterator, Tuple

import kagglehub
import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from kagglehub import KaggleDatasetAdapter
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

from .utils import DATA_DIR, RANDOM_SEED, create_logger

warnings.simplefilter("ignore")
np.random.seed(RANDOM_SEED)
logger = create_logger(__name__)


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

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        data_home: str = DATA_DIR,
        random_state: int = RANDOM_SEED,
        use_additional_datasets: bool = True,
        test_size: float = 0.2,
        valid_to_test_size: float | None = 0.5,
    ):
        """
        Initialize the BinaryDatasetManager with the specified data home and random state.

        Args:
            data_home (str): Directory to store datasets.
            random_state (int): Random seed for reproducibility.
            use_additional_datasets (bool): Whether to include additional datasets.
            test_size (float): Proportion of the dataset to include in the test split.
            valid_to_test_size (float | None): Ratio of test to validation set size.
                If None, the test set is not split into validation and test sets and
                validation data is set to None.
        """
        logger.info("Fetching datasets from imbalanced-learn...")
        self.datasets = fetch_datasets(
            data_home=data_home,
            random_state=random_state,
            shuffle=True,
            verbose=True,
        )

        if use_additional_datasets:
            logger.info("Fetching additional datasets...")
            self.datasets.update(self._fetch_additional_datasets())

        logger.info("Preparing datasets...")
        self._prepare_datasets()

        self.test_size = test_size
        self.valid_to_test_size = valid_to_test_size
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
                Tuple[np.ndarray | None, np.ndarray | None],
                Tuple[np.ndarray, np.ndarray],
            ],
        ]
    ]:
        """
        Serve the datasets for training and testing.
        This method splits the datasets into training and testing sets,
        performs preprocessing, and applies oversampling if specified.

        Args:
            random_state (int): Random seed for reproducibility.
            oversampler (Any | None): Oversampling method to apply to the training set.
            preprocessing_pipeline_creator (Callable[[np.ndarray], ColumnTransformer] | None):
                Function to create a preprocessing pipeline that takes a feature matrix and returns
                a ColumnTransformer.
        Yields:
                ...: Dataset name and a tuple containing training, validation (if applicable),
                and testing data.
        Raises:
            ValueError: If the dataset is not binary.
        """
        if preprocessing_pipeline_creator is None:
            preprocessing_pipeline_creator = self.get_preprocessing_pipeline

        for dataset_name, (X, y) in self.df[["data", "target"]].iterrows():
            logger.debug("Serving dataset: %s", dataset_name)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, stratify=y, random_state=random_state
            )

            if self.valid_to_test_size is not None:
                X_test, X_val, y_test, y_val = train_test_split(
                    X_test,
                    y_test,
                    test_size=self.valid_to_test_size,
                    stratify=y_test,
                    random_state=random_state,
                )
            else:
                X_val = y_val = None

            logger.debug(
                "Train shape: %s, Valid shape: %s, Test shape: %s",
                str(X_train.shape),
                str(X_val.shape) if X_val is not None else "None",
                str(X_test.shape),
            )

            pipeline = preprocessing_pipeline_creator(X_train)
            X_train = pipeline.fit_transform(X_train)
            X_test = pipeline.transform(X_test)
            if X_val:
                X_val = pipeline.transform(X_val)

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
                    (X_val, y_val),
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
