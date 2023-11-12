import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

from utils import setupLogger, plot_boxplots, plot_histograms, plot_class_histograms,seperateDataByEpitope, get_best_f1
from data_preprocessing import TCRContrastiveDataset

from modelBYOL import SiameseNetworkBYOL as SiameseNetwork, evaluate_model
#from modelContrastive import SiameseNetwork, evaluate_model


def main():
    model_name = "model_21.pt"
    model_path = "../output/byolModel/"+model_name
    output_dir = f"../output/byolModel/{model_name[:-3]}/"
    # model_name = "model_20.pt"
    # model_path = "../output/contrastiveModel/"+model_name
    # output_dir = f"../output/contrastiveModel/{model_name[:-3]}/"

    logger = setupLogger(output_dir+"output.txt")
    test_data = TCRContrastiveDataset.load('../output/test_dataset_contrastive.pickle')
    test_loader = DataLoader(test_data, batch_size=10000, num_workers=6, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(test_data.tensor_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    labels, distances = evaluate_model(test_loader, model, device)

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

    plot_boxplots(similar_dists, dissim_dists)
    plt.show()

    resWD = wasserstein_distance(similar_dists,dissim_dists)
    logger.info('\n== Wasserstein distances ==')
    logger.info(f'{"All":10} : {resWD}')

    dist_dict = seperateDataByEpitope(test_data, test_data.epitopes, distances)
    avg_score = 0
    for epitope in test_data.epitopes:
        pdist = dist_dict[epitope][1]
        ndist = dist_dict[epitope][0]
        score = wasserstein_distance(pdist, ndist)
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
