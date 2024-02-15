import torch
import timm.optim
from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import plot_losses
from modelBYOL import SiameseNetworkBYOL, BYOLLoss, evaluate_model, encode_data
from data_preprocessingBYOL import TCRDataset, ValDataset, Refset
from backbones import *





def classify(encodings, epitopes, ref_encodings, ref_epitopes, K=5):
    predictions = []

    print(f'\rFitting Nearest Neighbours', end='')
    nn = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    nn.fit(ref_encodings)

    print(f'\rCalculating Nearest Neighbours', end='')
    l_dist, l_indices = nn.kneighbors(encodings)

    print(f'\rParsing predictions', end='')
    for i, epitope in enumerate(epitopes):
        n_idx = l_indices[i]
        n_dist = l_dist[i]

        mx_dist = max(n_dist)
        n_dist = [(mx_dist+1 - d)/(mx_dist+1) for d in n_dist]

        matching_dists = [n_dist[i] for i, idx in enumerate(n_idx) if ref_epitopes[idx] == epitope]
        match_sum = sum(matching_dists)

        predictions.append(match_sum / len(n_dist))

    return predictions


def train(epochs, training_loader, validation_loader, net, criterion, optimizer, device):
    losses = []
    eval_scores = []

    reference_data = pd.DataFrame(training_loader.dataset.encoding_dict.values(), columns =['sequence', 'epitope'])
    print(f'\rLoading reference data', end='')
    ref_encodings = Refset(list(reference_data['sequence']))
    ref_loader = DataLoader(ref_encodings, batch_size=10000, num_workers=6, shuffle=False)
    print('\r'+' '*40, end='\r')

    for epoch in range(epochs):
        net.train()
        print(f"Epoch {epoch}")
        n_batches = len(training_loader)
        epoch_loss = 0
        for i, data in enumerate(training_loader, 0):
            print(f'\rBatch {i}/{n_batches}', end='')
            seq0, seq1 = data
            seq0, seq1 = seq0.to(device=device, dtype=torch.float), seq1.to(device=device, dtype=torch.float),

            optimizer.zero_grad()
            p1, z2, p2, z1 = net(seq0, seq1)

            loss = criterion(p1, z2, p2, z1)
            loss.backward()

            epoch_loss += loss.item()

            optimizer.step()

        print('\r', end='')
        epoch_loss /= n_batches
        print(f"Current loss {epoch_loss}")
        torch.save(net.state_dict(), f'../output/byolModel/model_{epoch}.pt')

        net.eval()

        print(f'\rEncoding validation data', end='')
        encodings, epitopes, labels = evaluate_model(validation_loader, net, device)

        print(f'\rEncoding reference data', end='')
        ref_encodings = encode_data(ref_loader, net, device)
        ref_epitopes = reference_data['epitope']

        print(f'\rClassifying validation data', end='')

        best_k = 1
        best_score = 0
        for k in range(1, 30):
            val_pred = classify(encodings, epitopes, ref_encodings, ref_epitopes, K=k)
            print('\r' + ' '*40, end='\r')

            results = pd.DataFrame.from_dict({'epitope': epitopes, 'y': labels, 'y_pred': val_pred})
            scores = []
            grouped_data = dict(tuple(results.groupby("epitope")))
            for epitope in list(grouped_data.keys()):
                data = grouped_data[epitope]
                y, y_pred = data['y'], data['y_pred']

                score = roc_auc_score(y, y_pred, max_fpr=0.1)
                scores.append(score)
            scores.extend([0.5, 0.5, 0.5, 0.5])
            macro_auc = np.average(scores)
            if macro_auc > best_score:
                best_score = macro_auc
                best_k = k

        y_pred = classify(encodings, epitopes, ref_encodings, ref_epitopes, K=best_k)
        print('\r' + ' '*40, end='\r')

        scores = []
        results = pd.DataFrame.from_dict({'epitope': epitopes, 'y': labels, 'y_pred': y_pred})
        grouped_data = dict(tuple(results.groupby("epitope")))

        for epitope in list(grouped_data.keys()):
            data = grouped_data[epitope]
            y, y_pred = data['y'], data['y_pred']

            score = roc_auc_score(y, y_pred, max_fpr=0.1)
            scores.append(score)

        scores.extend([0.5, 0.5, 0.5, 0.5])
        evalscore = np.average(scores)

        print(f"Current eval score {evalscore}\n")

        eval_scores.append(evalscore)
        losses.append(epoch_loss)

    return net, losses, eval_scores


def main():
    config = {
        "BatchSize": 4096,
        "Epochs": 24,
    }

    train_data = TCRDataset.load('../output/trainBYOL.pickle')
    validation_data = ValDataset.load('../output/valBYOL.pickle')

    input_size = train_data.tensor_size

    training_loader = DataLoader(train_data, batch_size=config['BatchSize'], shuffle=True, num_workers=12)
    test_loader = DataLoader(validation_data, batch_size=4096, num_workers=12, shuffle=False)

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