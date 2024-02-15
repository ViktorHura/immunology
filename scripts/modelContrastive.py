import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import *


class SiameseNetworkContrastive(nn.Module):
    def __init__(self, input_shape, backbone=None):
        super(SiameseNetworkContrastive, self).__init__()
        if backbone is None:
            backbone = ImRexBackbone(input_shape)
            # Setting up CNN Layers
        self.backbone = backbone

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.backbone(input1)
        # forward pass of input 2
        output2 = self.backbone(input2)

        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


def evaluate_model(test_loader, model, device):
    labels = []
    epitopes = []
    encodings = []

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            seq, epitope, label = data
            output = model.backbone(seq.to(device=device, dtype=torch.float))

            output = output.to("cpu")

            labels += label.tolist()
            encodings += output.tolist()
            epitopes += list(epitope)

    return encodings, epitopes, labels


def encode_data(data_loader, model, device):
    encodings = []
    with torch.no_grad():
        for i, seq in enumerate(data_loader, 0):
            output = model.backbone(seq.to(device=device, dtype=torch.float))

            output = output.to("cpu")
            encodings += output.tolist()

    return encodings
