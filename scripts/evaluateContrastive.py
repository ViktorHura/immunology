import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

from modelContrastive import SiameseNetwork, ContrastiveLoss
from data_preprocessing import TCRContrastiveDataset


def plot_curve(x_values, y_values, xlabel, ylabel):
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(x_values, y_values)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)


def main():
    test_data = TCRContrastiveDataset.load('../output/test_dataset_contrastive.pickle')
    test_loader = DataLoader(test_data, batch_size=10000, num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(test_data.tensor_size).to(device)
    model.load_state_dict(torch.load('../output/contrastiveModel/model_11.pt'))
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

    fpr, tpr, thresholds = roc_curve(labels, distances, drop_intermediate=True, pos_label=0)
    precision, recall, thresholds = precision_recall_curve(labels, distances, drop_intermediate=True, pos_label=0)

    plot_curve(fpr, tpr, "False Positive Rate", "True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    plot_curve(recall, precision, "Recall", "Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

    print(f"roc_auc {auc(fpr, tpr)}")


if __name__ == "__main__":
    main()
