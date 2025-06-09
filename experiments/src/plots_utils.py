"""Utility functions for plotting results from experiments."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PLOTS_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plots"))


# pylint: disable=too-many-arguments,too-many-positional-arguments
def create_results_bar_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    param: str,
    pallette: list,
    i: int = 0,
    target: str = "f1",
    y_label: str = "Mean F1 Score",
    bbox_right: float = 1.12,
    round_digits: int = 4,
) -> None:
    """
    Create a bar plot on the given axes for result data.

    This function generates a bar plot that compares F1 scores for different oversampling options.
    It sets up the color palette, handles the legend display based on the provided index, and adds
    bar labels with formatted values.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to draw the plot.
        df (pandas.DataFrame): DataFrame containing the results. It must include the columns
            'oversampling_option', 'f1', and 'oversampling_hue'.
        param (str): A string representing the parameters being visualized, used in the plot title.
        pallette (list): A list of colors to use for the bars in the plot.
        i (int, optional): Determines if the legend should be displayed.
            If i != 0, the legend is removed. Defaults to 0.
        target (str, optional): The target metric to plot. Defaults to "f1".
        y_label (str, optional): The label for the y-axis. Defaults to "Mean F1 Score".
        bbox_right (float, optional): The x-coordinate for the legend box anchor.
        round_digits (int, optional): The number of decimal places to
            round the bar labels. Defaults to 4.
    """
    sns.barplot(
        data=df,
        x="oversampling_option",
        y=target,
        hue="oversampling_hue",
        ax=ax,
        palette=pallette,
        capsize=0.2,
        order=sorted(df["oversampling_option"].unique()),
        hue_order=sorted(df["oversampling_hue"].unique()),
    )

    if i != 0:
        ax.legend_.remove()
    else:
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

        ax.legend(
            title="Oversampling Method",
            bbox_to_anchor=(bbox_right, 1.0),
            loc="upper right",
            ncol=1,
            handles=handles,
            labels=labels,
        )

    for container in ax.containers:
        ax.bar_label(
            container, fmt=f"%.{round_digits}f", label_type="center", rotation=90
        )
    ax.set_title(f"Parameters: {param}", fontsize=10)
    ax.set_xlabel("Oversampling type")
    ax.set_ylabel(y_label)


def save_plot(filename: str, dpi: int = 300, bbox_inches: str = "tight") -> None:
    """
    Save the plot to a file.

    Args:
        filename (str): The name of the file to save the plot to.
        dpi (int, optional): Dots per inch for the saved figure. Defaults to 300.
        bbox_inches (str, optional): Bounding box in inches. Defaults to "tight".
    """
    path = os.path.join(PLOTS_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()
