"""Utility functions for training the experiments module."""

import logging
import os
import time
from typing import Any

RANDOM_SEED: int = 42

HOME: str = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
DATA_DIR: str = os.path.join(HOME, "data")
RESULTS_DIR: str = os.path.join(HOME, "results")


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
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
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


create_directories()
