"""Utility functions for training the experiments module."""

import os

HOME: str = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
DATA_DIR: str = os.path.join(HOME, "data")
RESULTS_DIR: str = os.path.join(HOME, "results")


def create_directories():
    """
    Create directories for data and results if they do not exist.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


create_directories()
