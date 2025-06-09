# pylint: disable=import-outside-toplevel
"""Experiments for smote + bagging"""

import logging
from functools import partial
from typing import Any, Dict, List

from src.inspire import MergingStrategy
from src.utils import PARAM_GRID as TREE_PARAM_GRID
from src.utils import TimedLogger, create_logger, get_parser

# TODO: define the search space for the inspire # pylint: disable=fixme
PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [100, 500],
    "enn_neighbors": [],
    "enn_min_matching_neighbors": [],
    "oversampling_neighbors": [],
    "val_to_train_neighbors": [],
    "step_size": [],
    "oversampling_ratio": [],
    "adaptive_oversampling_step": [True, False],
    "br_threshold": [],
    "bp_theta": [],
    "mask_merging_strategy": [MergingStrategy.AND, MergingStrategy.OR],
    "remove_outliers_": [True, False],
}


def main():
    """Main function to run the experiment."""
    parser = get_parser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--search-hyper-param",
        action="store_true",
        help="Search hyper parameters space for INSPIRE.",
    )
    group.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate INSPIRE against remaining models.",
    )
    args = parser.parse_args()

    from src.dataset_manager import BinaryDatasetManager
    from src.inspire import InspireClassifier
    from src.inspire_experiments import perform_experiment

    logger = create_logger(__name__)

    with TimedLogger("Creating dataset manager", logger=logger, level=logging.INFO):
        data_manager = BinaryDatasetManager(
            use_additional_datasets=not args.ignore_additional_datasets,
        )

    if args.evaluate:
        inspire_param_grid = {
            "n_estimators": PARAM_GRID["n_estimators"],
        }
        base_model_param_grid = TREE_PARAM_GRID
    else:
        inspire_param_grid = PARAM_GRID
        base_model_param_grid = {}

    # pylint: disable=duplicate-code
    with TimedLogger("Running experiment", logger=logger, level=logging.INFO):
        perform_experiment(
            experiment_name=args.experiment_name,
            base_model_param_grid=base_model_param_grid,
            inspire_param_grid=inspire_param_grid,
            model_class=partial(
                InspireClassifier, logging_level=args.log_level, approximate_knn_=False
            ),
            data_manager=data_manager,
            start=args.start,
            end=args.end,
            n_takes=args.n_takes,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
            postfix="evaluate" if args.evaluate else "search",
        )


if __name__ == "__main__":
    main()
