import torch.nn as nn
import torch.nn.functional as F


class ImRexBackbone(nn.Module):
    def __init__(self, input_shape):
        super(ImRexBackbone, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 128, (1, 3), padding="same"),
            nn.ReLU(inplace=True),
            nn.LazyBatchNorm2d(),

            nn.Conv2d(128, 64, (1, 3), padding="same"),

            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(inplace=True),
            nn.LazyBatchNorm2d(),

            nn.Conv2d(64, 128, (1, 3), padding="same"),
            nn.LazyBatchNorm2d(),

            nn.Conv2d(128, 64, (1, 3), padding="same"),

            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(inplace=True),
            nn.LazyBatchNorm2d(),

            nn.Flatten(),

            nn.LazyLinear(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.cnn(input)


class SiameseNetwork(nn.Module):
    def __init__(self, input_shape, backbone=None):
        super(SiameseNetwork, self).__init__()
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