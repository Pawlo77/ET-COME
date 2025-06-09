"""Experiment utility functions."""

import os
from enum import Enum
from functools import partial
from typing import Any, Dict, Tuple

from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE

# from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE
from sklearn.base import BaseEstimator
from tqdm import tqdm

from .bagging_classifier import BaggingClassifier
from .dataset_manager import BinaryDatasetManager
from .experiments_utils import SCORING, get_performed_runs
from .training_utils import ParamRunner
from .utils import RANDOM_SEED, RESULTS_DIR

OVERSAMPLING_KWARGS: Dict[str, Any] = {
    "sampling_strategy": "minority",
}
OVERSAMPLING_CLASSES: Tuple[Tuple[str, Any]] = (
    (
        "SMOTE",
        partial(
            SMOTE,
            **OVERSAMPLING_KWARGS,
        ),
    ),
    (
        "BorderlineSMOTE",
        partial(BorderlineSMOTE, **OVERSAMPLING_KWARGS, kind="borderline-1"),
    ),
    (
        "ADASYN",
        partial(
            ADASYN,
            **OVERSAMPLING_KWARGS,
        ),
    ),
    # ("KMeansSMOTE", partial(KMeansSMOTE, **OVERSAMPLING_KWARGS)),
    # ("SVMSMOTE", partial(SVMSMOTE, **OVERSAMPLING_KWARGS)),
)
OVERSAMPLING_NEIGHBORS: Tuple[int] = (3, 5, 7)


class OversamplingOptions(Enum):
    """
    Enum for bagging options.
    """

    BASIC = 1
    ADVANCED = 2
    DISABLED = 3


def _get_oversampler_class(
    oversampler_name: str,
    oversampler_class: BaseEstimator,
    n_neighbors: int,
) -> BaseEstimator:
    """
    Get the oversampler class.

    Args:
        oversampler_name (str): The name of the oversampler.
        oversampler_class (BaseEstimator): The class of the oversampler.
        n_neighbors (int): The number of neighbors for the oversampler.
    Returns:
        BaseEstimator: The class of the oversampler.
    """
    if "smote" in oversampler_name.lower():  # pylint: disable=magic-value-comparison
        return partial(
            oversampler_class,
            k_neighbors=n_neighbors,
        )
    return partial(
        oversampler_class,
        n_neighbors=n_neighbors,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,dangerous-default-value,invalid-name
def perform_experiment(
    experiment_name: str,
    model_name: str,
    model_class: BaseEstimator,
    param_grid: dict,
    data_manager: BinaryDatasetManager,
    results_dir: str = RESULTS_DIR,
    scoring: dict = SCORING,
    verbose: bool = True,
    n_takes: int = 5,
    n_jobs: int = -1,
    bagging_classifier_class: BaggingClassifier = BaggingClassifier,
    oversampling_neighbors: Tuple[int] = OVERSAMPLING_NEIGHBORS,
    oversampling_classes: Tuple[Tuple[str, Any]] = OVERSAMPLING_CLASSES,
    oversampling_option: OversamplingOptions = OversamplingOptions.BASIC,
    base_random_state: int = RANDOM_SEED,
    start: int = None,
    end: int = None,
):
    """
    Perform the experiment.

    Args:
        experiment_name (str): The name of the experiment.
        model_name (str): The name of the model.
        model_class (BaseEstimator): The class of the model.
        param_grid (dict): The parameter grid for the model.
        data_manager (BinaryDatasetManager): The data manager.
        results_dir (str): The directory to save the results.
        scoring (dict): The scoring metrics.
        verbose (bool): Whether to print progress.
        n_takes (int): Number of runs for each configuration.
        n_jobs (int): Number of jobs to run in parallel.
        bagging_classifier_class (BaggingClassifier): The bagging classifier class.
        oversampling_neighbors (Tuple[int]): The number of neighbors for oversampling.
        oversampling_classes (Tuple[Tuple[str, Any]]): The oversampling classes.
        oversampling_option (OversamplingOptions): The oversampling option.
        base_random_state (int): The base random state for reproducibility.
        start (int): The start index for the experiment.
        end (int): The end index for the experiment.
    """
    results_dir = os.path.join(results_dir, f"run_{experiment_name}")
    os.makedirs(results_dir, exist_ok=True)

    total_cases = (
        len(oversampling_classes)
        * len(oversampling_neighbors)
        * len(data_manager)
        * n_takes
    )
    performed_runs = get_performed_runs(results_dir=results_dir)

    i = 0
    progress_bar = None
    # pylint: disable=duplicate-code,too-many-nested-blocks
    for n_neighbors in oversampling_neighbors:
        for oversampler_name, oversampler_class in oversampling_classes:
            oversampler_class = _get_oversampler_class(
                oversampler_name,
                oversampler_class,
                n_neighbors,
            )
            for dataset_name, (
                (X_train, y_train),
                (X_val, y_val),
                (X_test, y_test),
            ) in data_manager.serve(
                oversampler=(
                    oversampler_class()
                    if oversampling_option == OversamplingOptions.BASIC
                    else None
                ),
            ):
                for take in range(1, n_takes + 1):
                    i += 1
                    if start is not None and i < start:
                        continue
                    if end is not None and i > end:
                        break

                    _id = (
                        f"{dataset_name}__{oversampler_name}__{model_name}"
                        + f"__{n_neighbors}__{take}"
                    )

                    if _id in performed_runs:
                        if progress_bar is not None:
                            progress_bar.update(1)
                            progress_bar.set_postfix(
                                {
                                    "info": (
                                        f"{oversampler_name} with {model_name} "
                                        f"on {dataset_name} take {take}"
                                    )
                                }
                            )
                            progress_bar.refresh()
                        continue

                    if progress_bar is None:
                        progress_bar = tqdm(
                            total=total_cases if end is None else end,
                            desc="Total progress",
                            initial=i - 1,
                        )

                    if oversampling_option == OversamplingOptions.ADVANCED:
                        bagging_classifier_class = partial(
                            bagging_classifier_class,
                            oversampler_class=oversampler_class,
                        )

                    runner = ParamRunner(
                        base_estimator_class=model_class,
                        bagging_classifier_class=bagging_classifier_class,
                        param_grid=param_grid,
                        scoring=scoring,
                        n_jobs=n_jobs,
                        random_state=base_random_state + take,
                    )

                    # pylint: disable=duplicate-code
                    runner.fit(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        X_test=X_test,
                        y_test=y_test,
                        verbose=verbose,
                    )

                    runner.results_["dataset_name"] = dataset_name
                    runner.results_["oversampler_name"] = oversampler_name
                    runner.results_["model_name"] = model_name
                    runner.results_["n_neighbors"] = n_neighbors
                    runner.results_["take"] = take
                    runner.results_["oversampling_option"] = oversampling_option.name

                    runner.results_.to_json(
                        os.path.join(results_dir, "results.json"),
                        orient="records",
                        lines=True,
                        mode="a",
                    )
                    with open(
                        os.path.join(results_dir, "performed_runs.txt"),
                        mode="a",
                        encoding="utf-8",
                    ) as f:
                        f.write(f"{_id}\n")

                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "info": f"{model_name} ({oversampler_name}) on {dataset_name} ({take})"
                        }
                    )
                    progress_bar.refresh()
