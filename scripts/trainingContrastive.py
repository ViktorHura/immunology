import torch
import timm.optim
from torch.utils.data import DataLoader, Subset
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import plot_losses
from backbones import DenseBackbone
from modelContrastive import SiameseNetwork, ContrastiveLoss, evaluate_model
from data_preprocessing import TCRContrastiveDataset


def train(epochs, training_loader, validation_loader, net, criterion, optimizer, device):
    loss = []
    eval_scores = []

    for epoch in range(epochs):
        net.train()
        print(f"Epoch {epoch}")
        for i, data in enumerate(training_loader, 0):
            seq0, seq1, label, _ = data
            seq0, seq1, label = seq0.to(device=device, dtype=torch.float), seq1.to(device=device,
                                                                                   dtype=torch.float), label.to(
                device=device)
            optimizer.zero_grad()
            output1, output2 = net(seq0, seq1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

        print(f"Current loss {loss_contrastive.item()}")
        torch.save(net.state_dict(), f'../output/contrastiveModel/model_{epoch}.pt')

        net.eval()

        labels, distances = evaluate_model(validation_loader, net, device)

        evalscore = roc_auc_score(labels, np.negative(distances))

        print(f"Current eval score {evalscore}\n")

        eval_scores.append(evalscore)
        loss.append(loss_contrastive.item())

    return net, loss, eval_scores


def main():
    config = {
        "BatchSize": 4096,
        "Epochs": 48,
    }

    data = TCRContrastiveDataset.load('../output/training_dataset_contrastive.pickle')
    input_size = data.tensor_size

    validation_data = Subset(data, data.validation_indices)
    training_data = Subset(data, np.setdiff1d(np.arange(len(data)), data.validation_indices))

    training_loader = DataLoader(training_data, batch_size=config['BatchSize'], shuffle=True, num_workers=6)
    test_loader = DataLoader(validation_data, batch_size=10000, num_workers=6, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SiameseNetwork(input_size, backbone=DenseBackbone()).to(device)
    criterion = ContrastiveLoss()
    optimizer = timm.optim.Lars(net.parameters())

    model, losses, eval_scores = train(config['Epochs'], training_loader, test_loader, net, criterion, optimizer,
                                       device)

    plot_losses(config['Epochs'], losses)
    plt.savefig('../output/contrastiveModel/loss.png')
    plt.show()

    plot_losses(config['Epochs'], eval_scores, title="Evaluation Scores", ytitle="ROC AUC")
    plt.savefig('../output/contrastiveModel/eval.png')
    plt.show()

    with open('../output/contrastiveModel/results.json', 'w') as handle:
        json.dump({
            "config": config,
            "losses": losses,
            "evaluation_results": eval_scores
        }, handle, indent=4)


if __name__ == "__main__":
    main()
