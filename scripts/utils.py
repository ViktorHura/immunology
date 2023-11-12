import matplotlib.pyplot as plt
import numpy as np
import logging
import os


def plot_losses(epochs, losses, title="Training loss", xtitle="loss", ytitle="epoch"):
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


def get_best_f1(recall, precision, thresholds):
    f1_scores = 2 * recall * precision / (recall + precision)
    a_max = np.argmax(f1_scores)
    return np.max(f1_scores), thresholds[a_max], a_max


def plot_histograms(distances, sim, dissim, n_bins, title="Distance distributions"):
    fig, axs = plt.subplots(2, 1, sharex='col', tight_layout=True, figsize=([10, 4.8]))
    axs[0].hist(sim, bins=n_bins, weights=np.ones(len(sim)) / len(sim))
    axs[1].hist(dissim, bins=n_bins, weights=np.ones(len(dissim)) / len(dissim))

    fig.suptitle(title)
    axs[0].set_ylabel("Positive pair")
    axs[1].set_ylabel("Negative pair")
    #plt.xticks(np.arange(distances.min(), distances.max(), 0.5))
    plt.xlabel("Distance")


def seperateDataByEpitope(dataset, epitopes, distances):
    dist_dict = {}
    for epitope in epitopes:
        dist_dict[epitope] = ([], [])  # negative and positive pair distances

    for i, p in enumerate(dataset):
        _, _, typ, epitope_id = p
        dist_dict[epitopes[epitope_id]][typ].append(distances[i])
    return dist_dict


def plot_class_histograms(dataset, dist_dict, n_bins, output_dir, distances):
    for epitope in dataset.epitopes:
        pdist = dist_dict[epitope][1]
        ndist = dist_dict[epitope][0]

        plot_histograms(np.array(distances), pdist, ndist, n_bins, f"Distance distributions {epitope}")
        plt.savefig(output_dir + f"{epitope}_dist.png")
        plt.show()


def plot_boxplots(sim, dissim, title="Distance distributions"):
    fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=([10, 4.8]))
    axs.boxplot([sim, dissim])

    fig.suptitle(title)
    axs.set_xticklabels(["positive", "negative"])
    plt.ylabel("Distance")