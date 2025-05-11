"""Utility functions for reports."""

import os

import pandas as pd

from .utils import RESULTS_DIR


def read_results(run_id: int = 0, results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """
    Read the results from the specified run directory and return a DataFrame.

    Args:
        run_id (int, optional): The ID of the run to read results from.
        results_dir (str, optional): The directory where the results are stored.
    Returns:
        pd.DataFrame: A DataFrame containing the results of the specified run.
    """
    run_dir = os.path.join(results_dir, f"run_{run_id}")

    results_df = None
    for file in os.listdir(run_dir):
        if file.endswith(".pkl"):
            (dataset_name, smote_name, model_name, option, take) = os.path.basename(
                os.path.splitext(file)[0]
            ).split("__")
            cur_df = pd.read_json(
                os.path.join(run_dir, file), orient="records", lines=True
            )
            cur_df["dataset_name"] = dataset_name
            cur_df["smote_name"] = smote_name
            cur_df["model_name"] = model_name
            cur_df["option"] = option
            cur_df["take"] = take
            cur_df["run_id"] = run_id

            for stage in ["train", "test"]:
                for key in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
                    cur_df[f"{stage}_{key}"] = cur_df[stage].map(
                        lambda x, key=key: x[key]
                    )
            cur_df = cur_df.drop(columns=["train", "test"])

            if results_df is None:
                results_df = cur_df
            else:
                results_df = pd.concat([results_df, cur_df], ignore_index=True)

    return results_df
