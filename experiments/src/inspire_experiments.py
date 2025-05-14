"""Inspire utility functions."""

import os
from functools import partial

# from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from src.dataset_manager import BinaryDatasetManager
from src.training_utils import ParamRunner
from src.utils import RANDOM_SEED, RESULTS_DIR
from tqdm import tqdm

from .experiments_utils import SCORING, get_performed_runs


# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,dangerous-default-value,invalid-name
def perform_experiment(
    experiment_name: str,
    base_model_param_grid: dict,
    inspire_param_grid: dict,
    model_class: BaseEstimator,
    data_manager: BinaryDatasetManager,
    base_model_class: BaseEstimator = DecisionTreeClassifier,
    results_dir: str = RESULTS_DIR,
    scoring: dict = SCORING,
    verbose: bool = True,
    n_jobs: int = 1,
    n_takes: int = 5,
    base_random_state: int = RANDOM_SEED,
    start: int = None,
    end: int = None,
):
    """
    Perform the experiment.

    Args:
        experiment_name (str): The name of the experiment.
        base_model_param_grid (dict): The parameter grid for the model.
        inspire_param_grid (dict): The parameter grid for the inspire model.
        model_class (BaseEstimator): The class of the model.
        data_manager (BinaryDatasetManager): The data manager.
        base_model_class (BaseEstimator): The class of the base model to
            use for the inspire model.
        results_dir (str): The directory to save the results.
        scoring (dict): The scoring metrics.
        verbose (bool): Whether to print progress.
        n_jobs (int): Number of jobs to run in parallel.
        n_takes (int): Number of times to take the data.
        base_random_state (int): Base random state for reproducibility.
        start (int): Start index for the experiment.
        end (int): End index for the experiment.
    """
    results_dir = os.path.join(results_dir, f"run_{experiment_name}")
    os.makedirs(results_dir, exist_ok=True)

    base_model_param_list = ParameterGrid(base_model_param_grid)
    inspire_param_list = ParameterGrid(inspire_param_grid)
    performed_runs = get_performed_runs(results_dir=results_dir)
    total_cases = len(data_manager) * len(base_model_param_list) * n_takes

    i = 0
    progress_bar = None
    # pylint: disable=duplicate-code,too-many-nested-blocks
    for dataset_name, (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
    ) in data_manager.serve():
        for base_model_param in base_model_param_list:
            for take in range(n_takes):
                i += 1
                if start is not None and i < start:
                    continue
                if end is not None and i >= end:
                    break

                filename = f"{dataset_name}__inspire__{i}.json"
                if filename in performed_runs:
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.refresh()
                    continue

                if progress_bar is None:
                    progress_bar = tqdm(
                        total=total_cases if end is None else end,
                        desc="Total progress",
                        initial=i - 1,
                    )

                cur_model_class = partial(
                    model_class,
                    base_estimator=base_model_class(**base_model_param),
                )
                runner = ParamRunner(
                    base_estimator_class=cur_model_class,
                    param_list=inspire_param_list,
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

                runner.results_.to_json(
                    os.path.join(results_dir, filename),
                    orient="records",
                    lines=True,
                )

                progress_bar.update(1)
                progress_bar.refresh()
