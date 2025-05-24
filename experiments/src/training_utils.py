"""Utility functions for training."""

import multiprocessing as mp
import os
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from .utils import RANDOM_SEED, create_logger
import time

warnings.simplefilter("ignore")
np.random.seed(RANDOM_SEED)

logger = create_logger(__name__)


# pylint: disable=invalid-name,too-many-arguments
# pylint: disable=too-many-positional-arguments
def evaluate_wrapper(
    args: Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
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
                - X_val (np.ndarray | None): Validation features.
                - y_val (np.ndarray | None): Validation targets.
                - X_test (np.ndarray): Testing features.
                - y_test (np.ndarray): Testing targets.
        base_estimator_class (Callable[..., BaseEstimator]):
            Class of the base estimator to instantiate.
        scorers (Dict[str, Any]): Dictionary of scorer functions.
        random_state (int): Random seed for reproducibility.
        bagging_classifier_class (Callable[..., BaseEstimator] | None):
            Class of the bagging classifier to instantiate.
    Returns:
        Dict[str, Any]: Dictionary of evaluation results.
    """
    params, X_train, y_train, X_val, y_val, X_test, y_test = args
    # pylint: disable=duplicate-code
    return ParamRunner.evaluate(
        params=params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        base_estimator_class=base_estimator_class,
        scorers=scorers,
        random_state=random_state,
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
        scoring: Dict[str, str],
        param_grid: Dict[str, Any] | None = None,
        param_list: List[Dict[str, Any]] | None = None,
        random_state: int = RANDOM_SEED,
        bagging_classifier_class: Callable[..., BaseEstimator] | None = None,
        n_jobs: int = 1,
    ) -> None:
        """
        Initialize the ParamRunner.

        Args:
            base_estimator_class (Callable[..., BaseEstimator]): Class of the base estimator.
            scoring (Dict[str, str]): Dictionary of scoring metrics.
            param_grid (Dict[str, Any] | None): Dictionary representing the parameter grid.
            param_list (List[Dict[str, Any]] | None): List of parameter combinations.
            random_state (int, optional): Random seed for reproducibility.
            bagging_classifier_class (Callable[..., BaseEstimator] | None): Class of the
                bagging classifier to instantiate.
            n_jobs (int, optional): Number of jobs for parallel processing. Defaults to 1.
        """
        if (param_grid is None and param_list is None) or (
            param_grid is not None and param_list is not None
        ):
            raise RuntimeError("Either param_grid or param_list must be provided.")

        self.base_estimator_class = base_estimator_class
        self.bagging_classifier_class = bagging_classifier_class

        self.param_grid = param_grid
        self.param_list = param_list
        self.scoring = scoring

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.results_: pd.DataFrame | None = None
        self._scorers: Dict[str, Any] | None = None

        _dummy_iterations = 10**6
        _dummy_start = time.perf_counter()
        _dummy_sum = 0
        for i in range(_dummy_iterations):
            _dummy_sum += i
        self._baseline_ops = _dummy_iterations / (time.perf_counter() - _dummy_start)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> "ParamRunner":
        """
        Evaluate all parameter combinations on training and testing data.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_val (np.ndarray | None): Validation features.
            y_val (np.ndarray | None): Validation targets.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing targets.
            verbose (bool): Whether to show progress bar.
        Returns:
            ParamRunner: The fitted ParamRunner instance with results saved in results_.
        """
        self._scorers: Dict[str, Any] = {
            name: get_scorer(metric) for name, metric in self.scoring.items()
        }
        param_list: List[Dict[str, Any]] = self.param_list or list(
            ParameterGrid(self.param_grid)
        )

        # multiprocessing
        if self.n_jobs > 1 or self.n_jobs == -1:
            tasks: List[
                Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ] = [
                (params, X_train, y_train, X_val, y_val, X_test, y_test)
                for params in param_list
            ]
            results = self._run_parallel_evaluation(
                tasks=tasks,
                n_jobs=self.n_jobs,
                total=len(tasks),
                base_estimator_class=self.base_estimator_class,
                scorers=self._scorers,
                random_state=self.random_state,
                bagging_classifier_class=self.bagging_classifier_class,
                verbose=verbose,
            )
        # single process
        else:
            gen = tqdm(param_list, desc="Parameter Grid") if verbose else param_list
            results = [
                self.evaluate(
                    params=params,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    base_estimator_class=self.base_estimator_class,
                    scorers=self._scorers,
                    random_state=self.random_state,
                    bagging_classifier_class=self.bagging_classifier_class,
                )
                for params in gen
            ]

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
        base_estimator_class: Callable[..., BaseEstimator],
        scorers: Dict[str, Any],
        bagging_classifier_class: Callable[..., BaseEstimator] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        random_state: int = RANDOM_SEED,
    ) -> Dict[str, Any]:
        """
        Evaluate a single set of parameters with training and testing data.

        Args:
            params (Dict[str, Any]): Parameter set.
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing targets.
            base_estimator_class (Callable[..., BaseEstimator]):
                Class of the base estimator.
            scorers (Dict[str, Any]): Dictionary of scorer functions.
            random_state (int): Random seed for reproducibility.
            bagging_classifier_class (Callable[..., BaseEstimator] | None):
                Class of the bagging classifier to instantiate.
            X_val (np.ndarray | None): Validation features.
            y_val (np.ndarray | None): Validation targets.
            random_state (int): Random seed for reproducibility.
        Returns:
            Dict[str, Any]: Dictionary containing parameter configuration and scores.
        Raises:
            ValueError: If n_estimators is not in params for bagging.
        """
        # Instantiate model with or without bagging.
        if bagging_classifier_class:
            if "n_estimators" not in params:
                raise ValueError("n_estimators must be in params for bagging.")
            cur_params = params.copy()
            n_estimators = cur_params.pop("n_estimators")

            model = bagging_classifier_class(
                base_estimator=base_estimator_class(
                    **cur_params, random_state=random_state
                ),
                n_estimators=n_estimators,
                random_state=random_state,
            )
        else:
            model = base_estimator_class(**params, random_state=random_state)

        start_time = time.perf_counter()
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start_time

        # Compute an estimated number of baseline operations performed during the fit.
        estimated_operations = elapsed * ParamRunner._baseline_ops

        scores: Dict[str, Any] = {
            "params": params,
            "elapsed": elapsed,
            "estimated_operations": estimated_operations,
        }
        for mode, X, y in zip(["train", "test"], [X_train, X_test], [y_train, y_test]):
            mode_scores = {}
            for name, scorer in scorers.items():
                if name == "roc_auc":
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X)[:, 1]
                        mode_scores[name] = scorer._score_func(
                            y, y_proba
                        )  # pylint: disable=protected-access
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
        base_estimator_class: Callable[..., BaseEstimator],
        scorers: Dict[str, Any],
        random_state: int = RANDOM_SEED,
        bagging_classifier_class: Callable[..., BaseEstimator] | None = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run evaluations in parallel using multiprocessing.

        Args:
            tasks (List[Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
                A list of tasks where each task is a tuple with parameters and dataset splits.
            n_jobs (int): Number of processes to run in parallel.
            total (int): Total number of tasks.
            base_estimator_class (Callable[..., BaseEstimator]):
                Class of the base estimator.
            scorers (Dict[str, Any]): Dictionary of scorer functions.
            random_state (int): Random seed for reproducibility.
            bagging_classifier_class (Callable[..., BaseEstimator] | None):
                Class of the bagging classifier to instantiate.
            verbose (bool): Whether to show progress bar.
        Returns:
            List[Dict[str, Any]]: List of evaluation results.
        """
        processes = n_jobs if n_jobs != -1 else os.cpu_count()
        logger.debug("Using %d processes for parallel evaluation.", processes)

        with mp.Pool(processes=processes) as pool:
            func = partial(
                evaluate_wrapper,
                base_estimator_class=base_estimator_class,
                bagging_classifier_class=bagging_classifier_class,
                scorers=scorers,
                random_state=random_state,
            )
            results: List[Dict[str, Any]] = []

            gen = (
                tqdm(
                    pool.imap_unordered(func, tasks), total=total, desc="Parameter Grid"
                )
                if verbose
                else pool.imap_unordered(func, tasks)
            )
            for result in gen:
                results.append(result)

        return results
