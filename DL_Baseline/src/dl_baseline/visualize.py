import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_summary(all_results, datasets, models):
    heatmap_data = np.zeros((len(datasets), len(models)))

    for i, ds in enumerate(datasets):
        for j, model in enumerate(models):
            heatmap_data[i, j] = all_results[ds][model]["NASA_Score"]

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        xticklabels=models,
        yticklabels=datasets,
        cmap="viridis",
    )

    plt.title("NASA Score Heatmap (Lower = Better)")
    plt.show()
