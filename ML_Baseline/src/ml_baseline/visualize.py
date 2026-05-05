import numpy as np
import matplotlib.pyplot as plt


def plot_summary(all_results, datasets, models):
    phm = np.zeros((len(datasets), len(models)))
    rmse = np.zeros((len(datasets), len(models)))

    for i, ds in enumerate(datasets):
        for j, m in enumerate(models):
            phm[i, j] = all_results[ds][m]["test"]["phm08_score"]
            rmse[i, j] = all_results[ds][m]["test"]["rmse"]

    plt.figure()
    for j, m in enumerate(models):
        plt.plot(datasets, phm[:, j], marker="o", label=m)

    plt.title("PHM08 Score Comparison (Lower = Better)")
    plt.ylabel("PHM08 Score")
    plt.legend()
    plt.show()

    plt.figure()
    for j, m in enumerate(models):
        plt.plot(datasets, rmse[:, j], marker="o", label=m)

    plt.title("RMSE Comparison (Lower = Better)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

    x = np.arange(len(datasets))
    width = 0.25

    plt.figure()
    for j, m in enumerate(models):
        plt.bar(x + j * width, phm[:, j], width, label=m)

    plt.xticks(x + width, datasets)
    plt.title("PHM08 Score Overview")
    plt.ylabel("PHM08 Score")
    plt.legend()
    plt.show()
