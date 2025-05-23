# pylint: disable=import-outside-toplevel
"""Experiments for smote + bagging"""

import logging
from typing import Any, Dict, List

from sklearn.tree import DecisionTreeClassifier
from src.utils import PARAM_GRID as TREE_PARAM_GRID
from src.utils import TimedLogger, create_logger, get_parser

PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [100, 500],
    **TREE_PARAM_GRID,
}


def main():
    """Main function to run the experiment."""
    parser = get_parser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--base", action="store_true", help="Use the BASIC oversampling option."
    )
    group.add_argument(
        "--advanced", action="store_true", help="Use the ADVANCED oversampling option."
    )
    args = parser.parse_args()

    from src.bagging_experiments import OversamplingOptions, perform_experiment
    from src.dataset_manager import BinaryDatasetManager

    logger = create_logger(__name__)

    if args.base:
        oversampling_option = OversamplingOptions.BASIC
    else:
        oversampling_option = OversamplingOptions.ADVANCED

    with TimedLogger("Creating dataset manager", logger=logger, level=logging.INFO):
        # ignore_validation_datasets - they are not used yet we
        # want same splits as in the inspire experiments
        data_manager = BinaryDatasetManager(
            use_additional_datasets=not args.ignore_additional_datasets,
            only_additional_datasets=args.only_additional_datasets,
            only_large_datasets=args.only_large_datasets,
            only_small_datasets=args.only_small_datasets,
            ignore_validation_datasets=True,
        )

    # pylint: disable=duplicate-code
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
            n_takes=args.n_takes,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
