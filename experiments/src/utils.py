"""Utility functions for training the experiments module."""

import argparse
import logging
import os
import time
from typing import Any, Dict, List

RANDOM_SEED: int = 42

HOME: str = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
DATA_DIR: str = os.path.join(HOME, "data")
RESULTS_DIR: str = os.path.join(HOME, "results")


PARAM_GRID: Dict[str, List[Any]] = {
    "max_depth": [5, 10, 15],
    "min_samples_split": [3, 5, 8],
    "min_samples_leaf": [2, 3, 5],
    "criterion": ["gini", "entropy"],
    "max_features": [None, "sqrt"],
}

LOGGING_LEVEL: str = logging.INFO


class TimedLogger:
    """
    A context manager for measuring and logging the elapsed time for a block of code.
    """

    def __init__(self, msg: str, logger: logging.Logger, level: int = logging.DEBUG):
        """
        Initializes a TimedLogger instance.

        Args:
            msg (str): The base log message.
            logger (logging.Logger): The logger to use.
            level (int, optional): The logging level. Defaults to logging.DEBUG.
        """
        self.msg = msg
        self.level = level
        self.logger = logger

    def __enter__(self) -> "TimedLogger":
        self.start_time = time.time()  # pylint: disable=attribute-defined-outside-init
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        elapsed_time = time.time() - self.start_time
        self.logger.log(self.level, self.msg + f"... ({elapsed_time:.2f} seconds)")


def create_logger(name: str) -> logging.Logger:
    """
    Create a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The created logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)

    ch = logging.StreamHandler()
    ch.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def create_directories():
    """
    Create directories for data and results if they do not exist.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def get_parser() -> argparse.ArgumentParser:
    """Get the argument parser for the experiment."""
    parser = argparse.ArgumentParser(
        description="Perform experiments with different oversampling options."
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ignore-additional-datasets",
        action="store_true",
        help="Whether to use additional datasets.",
    )
    group.add_argument(
        "--only-additional-datasets",
        action="store_true",
        help="Use only additional datasets.",
    )
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument(
        "--only-large-datasets",
        action="store_true",
        help="Use only large datasets.",
    )
    group2.add_argument(
        "--only-small-datasets",
        action="store_true",
        help="Use only small datasets.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of jobs to run in parallel."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to print progress."
    )
    parser.add_argument(
        "--n-takes",
        type=int,
        default=5,
        help="Number of runs for each configuration.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level.")

    global LOGGING_LEVEL  # pylint: disable=global-statement
    LOGGING_LEVEL = logging.WARNING

    return parser


create_directories()
