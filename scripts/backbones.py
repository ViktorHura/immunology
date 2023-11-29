import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ImRexBackbone(nn.Module):
    def __init__(self, input_shape):
        super(ImRexBackbone, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 128, (2, 3), padding="same"),
            nn.ReLU(inplace=True),
            nn.LazyBatchNorm2d(),

            nn.Conv2d(128, 64, (2, 3), padding="same"),

            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(inplace=True),
            nn.LazyBatchNorm2d(),

            nn.Conv2d(64, 128, (2, 3), padding="same"),
            nn.LazyBatchNorm2d(),

            nn.Conv2d(128, 64, (2, 3), padding="same"),

            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(inplace=True),
            nn.LazyBatchNorm2d(),

            nn.Flatten(),

            nn.LazyLinear(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.cnn(input)
