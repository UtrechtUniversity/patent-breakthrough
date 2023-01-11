"""Functions to visualize the results of document embeddings."""

from matplotlib import pyplot as plt


def plot_cpc_correlations(years, correlations):
    """Plot correlation of embeddings with CPC classifications."""
    for model_name, cor in correlations.items():
        plt.plot(years, cor, label=model_name)
    plt.legend()
