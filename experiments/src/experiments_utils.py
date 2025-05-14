"""Scoring metrics for model evaluation."""

import os
from typing import Dict, Set

SCORING: Dict[str, str] = {
    "accuracy": "accuracy",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
}


def get_performed_runs(results_dir: int) -> Set[str]:
    """Get the performed runs."""
    return set(os.listdir(results_dir))
