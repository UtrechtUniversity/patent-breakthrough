"""Functions to visualize the results of document embeddings."""
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from collections import defaultdict


def plot_cpc_correlations(correlations):
    """Plot correlation of embeddings with CPC classifications."""
    plt.figure(dpi=100)
    for model_name, cor_data in correlations.items():
        plt.plot(cor_data["year"], cor_data["correlations"], label=model_name)
    plt.legend()


def plot_window_difference(analysis, window_1, window_2, impact=True, show_max=100,
                           model=None):
    results_1 = defaultdict(list)
    results_2 = defaultdict(list)
    for window, model in tqdm(analysis.data.iterate_window_models(model_name=model)):
        cur_impact_1, cur_novelty_1, _ = analysis._compute_impact_novelty(
            window, model, window=(1, 5))
        cur_impact_2, cur_novelty_2, _ = analysis._compute_impact_novelty(
            window, model, window=(6, 10))
        if impact:
            results_1[model].extend(cur_impact_1)
            results_2[model].extend(cur_impact_2)
        else:
            results_1[model].extend(cur_novelty_1)
            results_2[model].extend(cur_novelty_2)
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
