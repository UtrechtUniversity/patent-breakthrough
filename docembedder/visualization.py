"""Functions to visualize the results of document embeddings."""

from __future__ import annotations
from typing import Optional, Union
from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from docembedder.analysis import DocAnalysis


def plot_cpc_correlations(correlations):
    """Plot correlation of embeddings with CPC classifications."""
    plt.figure(dpi=100)
    for model_name, cor_data in correlations.items():
        plt.plot(cor_data["year"], cor_data["correlations"], label=model_name)
    plt.legend()


def plot_window_difference(  # pylint: disable=too-many-arguments,too-many-locals
        analysis: DocAnalysis,
        window_1: Optional[Union[int, tuple[int, int]]],
        window_2: Optional[Union[int, tuple[int, int]]],
        impact: bool=True,
        show_max: int = 100,
        model_name: Optional[str]=None):
    """Plot the effect on novelty/impact of having a different window size.

    Arguments
    ---------
    analysis:
        Analysis object to get the data from.
    window_1:
        First window to compare.
    window_2:
        Second window to compare.
    impact:
        If true, plot the impacts, if false, plot the novelties.
    show_max:
        Maximum number of points to show in the plot. Doesn't affect the
        correlation.
    model:
        Model name to plot. If None, plot all available models in the same
        plot.
    """
    exponent = 1.0
    results_1: dict[str, list] = defaultdict(list)
    results_2: dict[str, list] = defaultdict(list)
    for window, model in tqdm(analysis.data.iterate_window_models(model_name=model_name)):
        imp_res_1 = analysis.compute_impact_novelty(
            window, model, window=(1, 5), exponents=[exponent])
        imp_res_2 = analysis.compute_impact_novelty(
            window, model, window=(6, 10))
        if impact:
            results_1[model].extend(imp_res_1[exponent]["impact"])
            results_2[model].extend(imp_res_1[exponent]["novelty"])
        else:
            results_1[model].extend(imp_res_2[exponent]["impact"])
            results_2[model].extend(imp_res_2[exponent]["novelty"])
    plt.figure(dpi=100)
    for model in results_1:
        result_r = spearmanr(results_1[model], results_2[model])
        if len(results_1[model]) > show_max:
            rand_idx = np.random.choice(len(results_1[model]), show_max, replace=False)
            res_1 = np.array(results_1[model])[rand_idx]
            res_2 = np.array(results_2[model])[rand_idx]
        else:
            res_1 = results_1[model]
            res_2 = results_2[model]
        plt.scatter(res_1, res_2, label=f"{model}: {result_r.correlation:.3f}")
        plt.xlabel(f"{window_1}")
        plt.ylabel(str(window_2))
    if impact:
        plt.title("Impact")
    else:
        plt.title("Novelty")
    plt.legend()
    plt.show()
