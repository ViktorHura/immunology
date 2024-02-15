import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import *


class SiameseNetworkBYOL(nn.Module):
    def __init__(self, input_shape, backbone_q=None, backbone_k=None, pred_dim=32, dim=256, m=0.996):
        super(SiameseNetworkBYOL, self).__init__()
        if backbone_q is None:
            backbone_q = ImRexBackbone(input_shape)
        if backbone_k is None:
            backbone_k = ImRexBackbone(input_shape)
        # Setting up CNN Layers
        self.encoder_q = backbone_q
        self.encoder_k = backbone_k
        self.m = m

        self.predictor = nn.Sequential(nn.Linear(pred_dim, dim),
                                       nn.BatchNorm1d(dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(dim, pred_dim))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, input1, input2):
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

        p1 = self.predictor(self.encoder_q(input1))  # NxC
        z2 = self.encoder_k(input2)  # NxC

        p2 = self.predictor(self.encoder_q(input2))  # NxC
        z1 = self.encoder_k(input1)  # NxC

        return p1, z2.detach(), p2, z1.detach()


class BYOLLoss(nn.Module):
    def __init__(self):
        super(BYOLLoss, self).__init__()

    def forward(self, p1, z2, p2, z1):
        x1 = F.normalize(p1, dim=-1, p=2)
        y1 = F.normalize(z2, dim=-1, p=2)
        loss1 = 2 - 2 * (x1 * y1).sum(dim=-1)

        x2 = F.normalize(p2, dim=-1, p=2)
        y2 = F.normalize(z1, dim=-1, p=2)
        loss2 = 2 - 2 * (x2 * y2).sum(dim=-1)

        loss = loss1 + loss2
        return loss.mean()


def evaluate_model(test_loader, model, device):
    labels = []
    epitopes = []
    encodings = []

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            seq, epitope, label = data
            output = model.encoder_q(seq.to(device=device, dtype=torch.float))

            output = output.to("cpu")

            labels += label.tolist()
            encodings += output.tolist()
            epitopes += list(epitope)

    return encodings, epitopes, labels


def encode_data(data_loader, model, device):
    encodings = []
    with torch.no_grad():
        for i, seq in enumerate(data_loader, 0):
            output = model.encoder_q(seq.to(device=device, dtype=torch.float))

            output = output.to("cpu")
            encodings += output.tolist()

    return encodings