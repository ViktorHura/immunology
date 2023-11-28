import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from utils import setupLogger, seperateDataByEpitope, plot_distances
from data_preprocessing import TCRDataset
from backbones import DenseBackbone, TransformerBackbone, BytenetEncoder

from modelBYOL import SiameseNetworkBYOL as SiameseNetwork, evaluate_model
# from modelContrastive import SiameseNetwork, evaluate_model


def main():
    model_name = "byte1/model_9.pt"
    model_path = "../output/byolModel/"+model_name
    output_dir = f"../output/byolModel/{model_name[:-3]}/"
    # model_name = "model_45.pt"
    # model_path = "../output/contrastiveModel/"+model_name
    # output_dir = f"../output/contrastiveModel/{model_name[:-3]}/"

    logger = setupLogger(output_dir+"output.txt")
    test_data = TCRDataset.load('../output/test_dataset_contrastive.pickle')
    test_loader = DataLoader(test_data, batch_size=10000, num_workers=6, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = SiameseNetwork(test_data.tensor_size, backbone_q=DenseBackbone(), backbone_k=DenseBackbone(), pred_dim=128).to(device)
    # model = SiameseNetwork(test_data.tensor_size, backbone=DenseBackbone()).to(device)
    model = SiameseNetwork(test_data.tensor_size,backbone_q=BytenetEncoder(test_data.tensor_size),
                             backbone_k=BytenetEncoder(test_data.tensor_size),).to(device)
    # model = SiameseNetwork(test_data.tensor_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    labels, distances = evaluate_model(test_loader, model, device)

    n_bins = math.ceil(math.sqrt(len(distances)))
    scale = (0, np.array(distances).max())

    similar_dists = [d for i, d in enumerate(distances) if labels[i] == 1]
    dissim_dists = [d for i, d in enumerate(distances) if labels[i] == 0]

    score = roc_auc_score(labels, np.negative(distances))

    plot_distances(similar_dists, dissim_dists, n_bins, scale, title=f"Distance distributions, ROC AUC = {score:.4f}")
    plt.savefig(output_dir + "ALL_dist.png")
    plt.show()

    logger.info('\n== ROC AUC Score ==')
    logger.info(f'{"All":10} : {score}')

    avg = 0
    dist_dict = seperateDataByEpitope(test_data, test_data.epitopes, distances)
    for epitope in test_data.epitopes:
        pdist = dist_dict[epitope][1]
        ndist = dist_dict[epitope][0]
        scores = pdist + ndist
        y_true = [1]*len(pdist) + [0]*len(ndist)

        score = roc_auc_score(y_true, np.negative(scores))
        avg += score

        plot_distances(pdist, ndist, n_bins, scale, title=f"Distance distributions {epitope}, ROC AUC = {score:.4f}")
        plt.savefig(output_dir + f"{epitope}.png")
        plt.show()

        logger.info(f'{epitope:10} : {score}')

    avg /= len(test_data.epitopes)
    logger.info(f'Macro AUC : {avg}')


if __name__ == "__main__":
    main()
