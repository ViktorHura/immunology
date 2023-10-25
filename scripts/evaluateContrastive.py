import numpy as np
import os
import logging
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import math

from modelContrastive import SiameseNetwork
from data_preprocessing import TCRContrastiveDataset


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
    plt.xticks(np.arange(distances.min(), distances.max(), 0.5))
    plt.xlabel("Distance")


def seperateDataByEpitope(dataset, distances):
    dist_dict = {}
    for epitope in dataset.epitopes:
        dist_dict[epitope] = ([], [])  # negative and positive pair distances

    for i, p in enumerate(dataset.pairs):
        _, _, typ, epitope_id = p
        dist_dict[dataset.epitopes[epitope_id]][typ].append(distances[i])
    return dist_dict


def plot_class_histograms(dataset, dist_dict, n_bins, output_dir, distances):
    for epitope in dataset.epitopes:
        pdist = dist_dict[epitope][1]
        ndist = dist_dict[epitope][0]

        plot_histograms(np.array(distances), pdist, ndist, n_bins, f"Distance distributions {epitope}")
        plt.savefig(output_dir + f"{epitope}_dist.png")
        plt.show()


def main():
    model_name = "model_168.pt"
    output_dir = f"../output/contrastiveModel/third/{model_name[:-3]}/"
    logger = setupLogger(output_dir+"output.txt")

    test_data = TCRContrastiveDataset.load('../output/test_dataset_contrastive.pickle')
    test_loader = DataLoader(test_data, batch_size=10000, num_workers=6, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(test_data.tensor_size).to(device)
    model.load_state_dict(torch.load('../output/contrastiveModel/third/'+model_name))
    model.eval()

    labels = []
    distances = []
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            seqA, seqB, label = data
            outputA, outputB = model(seqA.to(device=device, dtype=torch.float), seqB.to(device=device, dtype=torch.float))

            dist = F.pairwise_distance(outputA, outputB)
            labels += label.tolist()
            distances += dist.tolist()

    RocCurveDisplay.from_predictions(labels, distances, pos_label=0)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.savefig(output_dir+"roc.png")
    plt.show()

    PrecisionRecallDisplay.from_predictions(labels, distances, pos_label=0)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.savefig(output_dir + "prd.png")
    plt.show()

    precision, recall, thresholds = precision_recall_curve(labels, distances, pos_label=0)

    f1, t, idx = get_best_f1(precision, recall, thresholds)

    logger.info(f"best f1 score of\n\t{f1}\n\twith a threshold of {t}")
    logger.info(f"\twith recall of {recall[idx]}\n\twith precision of {precision[idx]}")

    predicted = np.where(distances > t, 0, 1)
    conf = confusion_matrix(labels, predicted, labels=[0, 1])
    logger.info("== confusion matrix ==")
    logger.info("dissimilar / similar")
    logger.info(conf)


    similar_dists = [d for i, d in enumerate(distances) if labels[i] == 1]
    dissim_dists = [d for i, d in enumerate(distances) if labels[i] == 0]

    resKS = ks_2samp(data1=similar_dists, data2=dissim_dists, alternative='two-sided', method='exact')
    logger.info('\n== Kolmogorov Smirnov p-values ==')
    logger.info(f'{"All":10} : {-math.log(resKS.pvalue)}')

    dist_dict = seperateDataByEpitope(test_data, distances)
    avg_score = 0
    for epitope in test_data.epitopes:
        pdist = dist_dict[epitope][1]
        ndist = dist_dict[epitope][0]
        resKS = ks_2samp(data1=pdist, data2=ndist, alternative='two-sided', method='exact')
        score = -math.log(resKS.pvalue)
        avg_score += score
        logger.info(f'{epitope:10} : {score}')
    avg_score /= len(test_data.epitopes)
    logger.info(f'{"Average":10} : {avg_score}')


    plot_histograms(np.array(distances), similar_dists, dissim_dists, 100)
    plt.savefig(output_dir + "ALL_dist.png")
    plt.show()

    plot_class_histograms(test_data, dist_dict, 100, output_dir, distances)


if __name__ == "__main__":
    main()
