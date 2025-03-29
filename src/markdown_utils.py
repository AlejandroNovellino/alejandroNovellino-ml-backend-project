"""
Utility functions for Markdown elements.
"""

import numpy as np
from IPython.core.display_functions import display
from IPython.display import Markdown


def show_comparison_table(metric_names: list[str], default_metrics: list[float],
                          optimized_metrics: list[float]) -> None:
    """
    Creates and show a Markdown table comparing default and optimized model metrics.

    Args:
        metric_names (list[str]): List of metric names.
        default_metrics (list[float]): List of metric values for the default model.
        optimized_metrics (list[float]): List of metric values for the optimized model.

    Returns:
        str: A Markdown table as a string.

    Raise:
        Exception: the metrics length is different from the default or optimized metrics.
    """

    if len(metric_names) != len(default_metrics) or len(metric_names) != len(optimized_metrics):
        raise Exception("Error: Metric lists must have the same length.")

    markdown_table = "| Metric | Default Model | Optimized Model |\n"
    markdown_table += "|---|---|---|\n"

    for i in range(len(metric_names)):
        markdown_table += f"| {metric_names[i]} | {np.round(default_metrics[i], 2)} | {np.round(optimized_metrics[i], 2)} |\n"

    # display the table
    display(Markdown(markdown_table))

    return None
