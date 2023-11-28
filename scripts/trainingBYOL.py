import torch
import timm.optim
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
import json
import matplotlib.pyplot as plt
import pandas as pd

from utils import plot_losses
from modelBYOL import SiameseNetworkBYOL, BYOLLoss, evaluate_model, encode_data
from data_preprocessing import TCRDataset, ValDataset
from backbones import *


def classify(encodings, epitopes, reference_data, model, device, K=5):
    predictions = []

    ref_encodings = TensorDataset(reference_data['sequence'])
    ref_loader = DataLoader(ref_encodings, batch_size=10000, num_workers=6, shuffle=False)
    ref_encodings = encode_data(ref_loader, model, device)

    ref_epitopes = reference_data['epitope']

    nn = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    nn.fit(ref_encodings)

    _, l_indices = nn.kneighbors(encodings)

    for i, epitope in enumerate(epitopes):
        n_idx = l_indices[i]

        n_epitopes = [ref_epitopes[x] for x in n_idx]
        matching = n_epitopes.count(epitope)

        predictions.append(matching / len(n_epitopes))

    return predictions


def train(epochs, training_loader, validation_loader, net, criterion, optimizer, device):
    losses = []
    eval_scores = []

    reference_data = pd.DataFrame(training_loader.dataset.encoding_dict.values(), columns =['sequence', 'epitope'])

    for epoch in range(epochs):
        net.train()
        print(f"Epoch {epoch}")
        n_batches = len(training_loader)
        for i, data in enumerate(training_loader, 0):
            print(f'\rBatch {i}/{n_batches}', end='')
            seq0, seq1 = data
            seq0, seq1 = seq0.to(device=device, dtype=torch.float), seq1.to(device=device, dtype=torch.float),

            optimizer.zero_grad()
            p1, z2, p2, z1 = net(seq0, seq1)

            loss = criterion(p1, z2, p2, z1)
            loss.backward()

            optimizer.step()

        print('\r', end='')
        print(f"Current loss {loss.item()}")
        torch.save(net.state_dict(), f'../output/byolModel/model_{epoch}.pt')

        net.eval()

        encodings, epitopes, labels = evaluate_model(validation_loader, net, device)
        y_pred = classify(encodings, epitopes, reference_data, net, device)

        evalscore = roc_auc_score(labels, y_pred)

        print(f"Current eval score {evalscore}\n")

        eval_scores.append(evalscore)
        losses.append(loss.detatch().item())

    return net, losses, eval_scores


def main():
    config = {
        "BatchSize": 4096,
        "Epochs": 1,
    }

    train_data = TCRDataset.load('../output/train.pickle')
    validation_data = ValDataset.load('../output/val.pickle')

    input_size = train_data.tensor_size

    training_loader = DataLoader(train_data, batch_size=config['BatchSize'], shuffle=True, num_workers=6)
    test_loader = DataLoader(validation_data, batch_size=6000, num_workers=6, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SiameseNetworkBYOL(input_size).to(device)
    criterion = BYOLLoss()
    optimizer = timm.optim.Lars(net.parameters())

    model, losses, eval_scores = train(config['Epochs'], training_loader, test_loader, net, criterion, optimizer,device)

    plot_losses(config['Epochs'], losses)
    plt.savefig('../output/byolModel/loss.png')
    plt.show()

    plot_losses(config['Epochs'], eval_scores, title="Evaluation Scores", ytitle="ROC AUC")
    plt.savefig('../output/byolModel/eval.png')
    plt.show()

    with open('../output/byolModel/results.json', 'w') as handle:
        json.dump({
            "config": config,
            "losses": losses,
            "evaluation_results": eval_scores
        }, handle, indent=4)


if __name__ == "__main__":
    main()
