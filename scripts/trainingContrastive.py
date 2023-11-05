import torch
from torch.utils.data import DataLoader, Subset
import json
import matplotlib.pyplot as plt
import numpy as np
from evaluateContrastive import evaluate_model, seperateDataByEpitope
from scipy.stats import wasserstein_distance

from modelContrastive import SiameseNetwork, ContrastiveLoss
from data_preprocessing import TCRContrastiveDataset


def train(epochs, dataloader, net, criterion, optimizer, device):
    loss = []

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            seq0, seq1, label, _ = data
            seq0, seq1, label = seq0.to(device=device, dtype=torch.float), seq1.to(device=device, dtype=torch.float), label.to(device=device)
            optimizer.zero_grad()
            output1, output2 = net(seq0, seq1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

        print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        loss.append(loss_contrastive.item())
        torch.save(net.state_dict(), f'../output/contrastiveModel/model_{epoch}.pt')

    return net, loss


def plot_losses(epochs, losses):
    x = np.array(range(epochs))
    y = np.array(losses)

    plt.plot(x, y)

    plt.title("Training loss")

    plt.ylabel("loss")
    plt.xlabel("epoch")


def main():
    config = {
        "BatchSize": 4096,
        "Epochs": 2,
	    "LR":0.001
    }

    data = TCRContrastiveDataset.load('../output/training_dataset_contrastive.pickle')
    input_size = data.tensor_size

    validation_data = Subset(data, data.validation_indices)
    training_data = Subset(data, np.setdiff1d(np.arange(len(data)), data.validation_indices))


    #training_data = torch.utils.data.Subset(training_data, range(1024))
    training_loader = DataLoader(training_data, batch_size=config['BatchSize'], shuffle=True, num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SiameseNetwork(input_size).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['LR'])

    model, losses = train(config['Epochs'], training_loader, net, criterion, optimizer, device)
    torch.save(model.state_dict(), '../output/contrastiveModel/model_final.pt')

    #print(losses)

    plot_losses(config['Epochs'], losses)
    plt.savefig('../output/contrastiveModel/loss.png')
    plt.show()

    test_loader = DataLoader(validation_data, batch_size=10000, num_workers=6, shuffle=False)
    model.eval()

    labels, distances = evaluate_model(test_loader, model, device)

    similar_dists = [d for i, d in enumerate(distances) if labels[i] == 1]
    dissim_dists = [d for i, d in enumerate(distances) if labels[i] == 0]

    resWD = wasserstein_distance(similar_dists, dissim_dists)
    print('\n== Wasserstein distances ==')
    print(f'{"All":10} : {resWD}')

    evaluation_results = {
        "all": resWD
    }

    dist_dict = seperateDataByEpitope(validation_data, data.epitopes, distances)
    avg_score = 0
    for epitope in data.epitopes:
        pdist = dist_dict[epitope][1]
        ndist = dist_dict[epitope][0]
        score = wasserstein_distance(pdist, ndist)
        evaluation_results[epitope] = score
        avg_score += score
        print(f'{epitope:10} : {score}')
    avg_score /= len(data.epitopes)
    evaluation_results["avg"] = avg_score
    print(f'{"Average":10} : {avg_score}')

    with open('../output/contrastiveModel/results.json', 'w') as handle:
        json.dump({
            "config": config,
            "losses": losses,
            "evaluation_results": evaluation_results
        }, handle, indent=4)


if __name__ == "__main__":
    main()
