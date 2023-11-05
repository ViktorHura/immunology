import copy
import torch
import torch.nn as nn
from modelContrastive import ImRexBackbone


class SiameseNetworkBYOL(nn.Module):
    def __init__(self, input_shape, backbone=None, pred_dim=32, dim=256, m=0.996):
        super(SiameseNetworkBYOL, self).__init__()
        if backbone is None:
            backbone = ImRexBackbone(input_shape)
        # Setting up CNN Layers
        self.encoder_q = backbone
        self.encoder_k = copy.deepcopy(backbone)
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

        return p1, p2, z1.detach(), z2.detach()