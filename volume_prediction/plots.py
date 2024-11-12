import sys
sys.path.append(r"C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip")
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def plot_error_and_confidence(error, confidence, bins):
    counts, bin_edges = np.histogram(error , bins=bins)
    bin_means = []
    for i in range(len(bin_edges) - 1):
        bin_values = confidence[(error >= bin_edges[i]) & (error < bin_edges[i+1])]
        bin_means.append(np.mean(bin_values) if len(bin_values) > 0 else np.nan)
    norm = mcolors.Normalize(vmin=min(bin_means), vmax=max(bin_means))
    cmap = plt.cm.Blues  

    # Plot histogram with color indicating mean values from data2
    plt.figure(figsize=(10, 6))
    for i in range(len(counts)):
        plt.bar(
            (bin_edges[i] + bin_edges[i+1]) / 2, counts[i],
            width=bin_edges[i+1] - bin_edges[i],
            color=cmap(norm(bin_means[i])),
            edgecolor='black'
        )
        if not np.isnan(bin_means[i]):
            plt.text(
                (bin_edges[i] + bin_edges[i+1]) / 2, counts[i] + 0.5,
                f"{bin_means[i]:.2f}",
                ha='center', va='bottom', color='black', fontsize=10
            )
    plt.show()

def plot_volume_vs_pred(volume, pred):
    plt.scatter(volume, pred)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel("Volume")
    plt.ylabel("Prediction")
    plt.show()