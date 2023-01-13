"""Functions to visualize the results of document embeddings."""

from matplotlib import pyplot as plt


def plot_cpc_correlations(correlations):
    """Plot correlation of embeddings with CPC classifications."""
    for model_name, cor_data in correlations.items():
        plt.plot(cor_data["year"], cor_data["correlations"], label=model_name)
    plt.legend()
