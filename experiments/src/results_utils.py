"""Utility functions to load and process results from experiments."""

import os
from typing import Tuple

import pandas as pd

RESULTS_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)


def load_results_small() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load results from a CSV file for advanced and base experiments.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing
            results for advanced and base experiments.
    """
    frames = [None, None]

    for i, name in enumerate(("advanced", "base")):
        # small non additional datasets
        for postfix in ("a", "b", "c"):
            results = pd.read_json(
                os.path.join(
                    RESULTS_DIR,
                    f"run_{name}_ignore_additional_datasets_small",
                    f"results_part_{postfix}.json",
                ),
                lines=True,
            )
            if postfix == "a":  # pylint: disable=magic-value-comparison
                results_all = results
            else:
                results_all = pd.concat([results_all, results], ignore_index=True)

        # small additional datasets
        results = pd.read_json(
            os.path.join(
                RESULTS_DIR,
                f"run_{name}_additional_datasets_small",
                "results.json",
            ),
            lines=True,
        )
        results_all = pd.concat([results_all, results], ignore_index=True)

        frames[i] = results_all

    return tuple(frames)


def create_results_df() -> pd.DataFrame:
    """
    Create a DataFrame with results for advanced and base experiments.

    Returns:
        pd.DataFrame: DataFrame containing results for both experiments.
    """
    advanced_results_small, base_results_small = load_results_small()

    tmp_df = pd.concat(
        [advanced_results_small, base_results_small],
        ignore_index=True,
    )

    original_rows = tmp_df.shape[0]
    tmp_df = tmp_df.dropna(subset=["test"]).reset_index(drop=True)
    print(
        f"Filtered out {original_rows - tmp_df.shape[0]} rows with missing test results."
    )

    test = pd.DataFrame(tmp_df["test"].values.tolist())
    results_df = pd.concat([tmp_df, test], axis=1)
    results_df = results_df.drop(columns=["train", "test"])

    return results_df
