import torch
from torch.utils.data import DataLoader, Subset
import json
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from modelContrastive import SiameseNetwork, ContrastiveLoss
from data_preprocessing import TCRContrastiveDataset


def train(epochs, dataloader, net, criterion, optimizer, device):
    loss = []

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            seq0, seq1, label = data
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
        "Epochs": 1,
	    "LR":0.0001
    }

    data = TCRContrastiveDataset.load('../output/training_dataset_contrastive.pickle')
    input_size = data.tensor_size

    validation_data = Subset(data, data.validation_indices)
    training_data = Subset(data, [i for i in range(len(data)) if i not in data.validation_indices])


    #training_data = torch.utils.data.Subset(training_data, range(1024))
    training_loader = DataLoader(training_data, batch_size=config['BatchSize'], shuffle=True, num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SiameseNetwork(input_size).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['LR'])

    model, losses = train(config['Epochs'], training_loader, net, criterion, optimizer, device)
    torch.save(model.state_dict(), '../output/contrastiveModel/model_final.pt')

    print(losses)

    with open('../output/contrastiveModel/results.json', 'w') as handle:
        json.dump({
            "config": config,
            "losses": losses
        }, handle, indent=4)

    plot_losses(config['Epochs'], losses)
    plt.savefig('../output/contrastiveModel/loss.png')
    plt.show()

    test_loader = DataLoader(validation_data, batch_size=10000, num_workers=6, shuffle=False)
    model.eval()

    labels = []
    distances = []
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            seqA, seqB, label = data
            outputA, outputB = model(seqA.to(device=device, dtype=torch.float),
                                     seqB.to(device=device, dtype=torch.float))

            dist = F.pairwise_distance(outputA, outputB)
            labels += label.tolist()
            distances += dist.tolist()


if __name__ == "__main__":
    main()
