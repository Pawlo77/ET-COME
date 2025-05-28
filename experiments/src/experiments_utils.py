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
    results_file = os.path.join(results_dir, "performed_runs.txt")
    if not os.path.exists(results_file):
        with open(results_file, mode="w", encoding="utf-8"):
            pass
        return set()

    with open(
        os.path.join(results_dir, "performed_runs.txt"), mode="r", encoding="utf-8"
    ) as f:
        performed_runs = f.read().splitlines()
    return set(performed_runs)
