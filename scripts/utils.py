import matplotlib.pyplot as plt
import numpy as np
import logging
import os


def plot_losses(epochs, losses, title="Training loss", ytitle="loss", xtitle="epoch"):
    x = np.array(range(epochs))
    y = np.array(losses)

    plt.plot(x, y)

    plt.title(title)
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)


def setupLogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = logging.FileHandler(path, mode="w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger


def plot_distances(sim, dissim, n_bins, scale, title="Distance distributions"):
    fig, axs = plt.subplots(3, 1, sharex='col', tight_layout=True, figsize=([20, 9.6]))
    plt.xlim(scale)
    axs[0].hist(sim, bins=n_bins, weights=np.ones(len(sim)) / len(sim))
    axs[1].hist(dissim, bins=n_bins, weights=np.ones(len(dissim)) / len(dissim))
    axs[2].boxplot([dissim, sim], vert=False)
    axs[2].set_yticklabels(["negative", "positive"])

    fig.suptitle(title, fontsize="xx-large", fontweight="bold")
    axs[0].set_ylabel("Positive pair")
    axs[1].set_ylabel("Negative pair")
    plt.xlabel("Distance")


def seperateDataByEpitope(dataset, epitopes, distances):
    dist_dict = {}
    for epitope in epitopes:
        dist_dict[epitope] = ([], [])  # negative and positive pair distances

    for i, p in enumerate(dataset):
        _, _, typ, epitope_id = p
        dist_dict[epitopes[epitope_id]][typ].append(distances[i])
    return dist_dict