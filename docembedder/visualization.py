from matplotlib import pyplot as plt


def plot_cpc_correlations(years, correlations):
    for model_name, cor in correlations.items():
        plt.plot(years, cor, label=model_name)
    plt.legend()
