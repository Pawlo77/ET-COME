"""Experiments for smote + bagging"""

import argparse
import logging
from typing import Any, Dict, List

from sklearn.tree import DecisionTreeClassifier
from src.dataset_manager import BinaryDatasetManager
from src.experiments_utils import OversamplingOptions, perform_experiment
from src.utils import TimedLogger, create_logger

logger = create_logger(__name__)

PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [100, 500],
    "max_depth": [5, 10, 15],
    "min_samples_split": [3, 5, 8],
    "min_samples_leaf": [2, 3, 5],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_features": [None, "sqrt"],
}


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(
        description="Perform experiments with different oversampling options."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--base", action="store_true", help="Use the BASIC oversampling option."
    )
    group.add_argument(
        "--advanced", action="store_true", help="Use the ADVANCED oversampling option."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index for the experiment (optional).",
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End index for the experiment (optional)."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name of the experiment.",
        default="0",
    )
    parser.add_argument(
        "--ignore-additional-datasets",
        action="store_true",
        help="Weather to use additional dataasets.",
    )
    args = parser.parse_args()

    if args.base:
        oversampling_option = OversamplingOptions.BASIC
    else:
        oversampling_option = OversamplingOptions.ADVANCED

    with TimedLogger("Creating dataset manager", logger=logger, level=logging.INFO):
        data_manager = BinaryDatasetManager(
            use_additional_datasets=not args.ignore_additional_datasets,
            valid_to_test_size=None,  # no validation set
        )

    with TimedLogger("Running experiment", logger=logger, level=logging.INFO):
        perform_experiment(
            start=args.start,
            end=args.end,
            experiment_name=args.experiment_name,
            model_name="DecisionTree",
            model_class=DecisionTreeClassifier,
            param_grid=PARAM_GRID,
            data_manager=data_manager,
            oversampling_option=oversampling_option,
        )


if __name__ == "__main__":
    main()
