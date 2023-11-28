import torch
import torch.nn as nn
import math
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


# class DenseBackbone(nn.Module):
#     def __init__(self):
#         super(DenseBackbone, self).__init__()
#         self.net = nn.Sequential(
#             nn.Flatten(),
#             nn.LazyLinear(512),
#             nn.ReLU(inplace=True),
#
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, input):
#         return self.net(input)
#
#
# class LSTMBackbone(nn.Module):
#     def __init__(self, input_shape, hidden_size=128, proj_size=32, layers=2):
#         super(LSTMBackbone, self).__init__()
#
#         self.batchnorm = nn.LazyBatchNorm2d()
#
#         self.lstm = nn.LSTM(input_size=input_shape[0], hidden_size=hidden_size, num_layers=layers,
#           batch_first=True, proj_size=proj_size)
#
#
#     def forward(self, input):
#         input = self.batchnorm(input)
#         input = torch.squeeze(input, len(input.shape)-2)
#         input = torch.transpose(input, len(input.shape)-1, len(input.shape)-2)
#         _, (_, out) = self.lstm(input)
#         out = out[:, -1, :]
#         return out
#
#
# class TransformerBackbone(nn.Module):
#     def __init__(self, input_shape, n_heads=8, n_encoder_layers=4):
#         super(TransformerBackbone, self).__init__()
#
#         self.pad = not(input_shape[0] % 2 == 0)
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_shape[0] + self.pad,
#             nhead=n_heads,
#             batch_first=True
#         )
#
#         self.encoder = nn.Sequential(
#             nn.TransformerEncoder(
#             encoder_layer=encoder_layer,
#             num_layers=n_encoder_layers,
#             norm=None
#             ),
#
#             nn.Flatten(),
#             nn.LazyLinear(32),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, input):
#         input = torch.squeeze(input, len(input.shape) - 2)
#
#         if self.pad:
#             input = nn.functional.pad(input, (0, 0, 0, 1), "constant", 0)
#
#         input = torch.transpose(input, len(input.shape) - 1, len(input.shape) - 2)
#         out = self.encoder(input)
#         return out
#
# ######################################
#
#
# def _same_pad(k=1, dil=1):
#     # assumes stride length of 1
#     # p = math.ceil((l - 1) * s - l + dil*(k - 1) + 1)
#     p = math.ceil(dil*(k - 1))
#     #print("padding:", p)
#     return p
#
#
# class ResBlock(nn.Module):
#     """
#         Note To Self:  using padding to "mask" the convolution is equivalent to
#         either centering the convolution (no mask) or skewing the convolution to
#         the left (mask).  Either way, we should end up with n timesteps.
#
#         Also note that "masked convolution" and "casual convolution" are two
#         names for the same thing.
#
#     Args:
#         d (int): size of inner track of network.
#         r (int): size of dilation
#         k (int): size of kernel in dilated convolution (odd numbers only)
#         casual (bool): determines how to pad the casual conv layer. See notes.
#     """
#     def __init__(self, d, r=1, k=3, casual=False, use_bias=False):
#         super(ResBlock, self).__init__()
#         self.d = d # input features
#         self.r = r # dilation size
#         self.k = k # "masked kernel size"
#         ub = use_bias
#         self.layernorm1 = nn.InstanceNorm1d(num_features=2*d, affine=True) # same as LayerNorm
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1x1_1 = nn.Conv1d(2*d, d, kernel_size=1, bias=ub) # output is "d"
#         self.layernorm2 = nn.InstanceNorm1d(num_features=d, affine=True)
#         self.relu2 = nn.ReLU(inplace=True)
#         if casual:
#             padding = (_same_pad(k,r), 0)
#         else:
#             p = _same_pad(k,r)
#             if p % 2 == 1:
#                 padding = [p // 2 + 1, p // 2]
#             else:
#                 padding = (p // 2, p // 2)
#         self.pad = nn.ConstantPad1d(padding, 0.)
#         #self.pad = nn.ReflectionPad1d(padding) # this might be better for audio
#         self.maskedconv1xk = nn.Conv1d(d, d, kernel_size=k, dilation=r, bias=ub)
#         self.layernorm3 = nn.InstanceNorm1d(num_features=d, affine=True)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.conv1x1_2 = nn.Conv1d(d, 2*d, kernel_size=1, bias=ub) # output is "2*d"
#
#     def forward(self, input):
#         x = input
#         x = self.layernorm1(x)
#         x = self.relu1(x)
#         x = self.conv1x1_1(x)
#         x = self.layernorm2(x)
#         x = self.relu2(x)
#         x = self.pad(x)
#         x = self.maskedconv1xk(x)
#         x = self.layernorm3(x)
#         x = self.relu3(x)
#         x = self.conv1x1_2(x)
#         #print("ResBlock:", x.size(), input.size())
#         x += input # add back in residual
#         return x
#
#
# class ResBlockSet(nn.Module):
#     """
#         The Bytenet encoder and decoder are made up of sets of residual blocks
#         with dilations of increasing size.  These sets are then stacked upon each
#         other to create the full network.
#     """
#     def __init__(self, d, max_r=16, k=3, casual=False):
#         super(ResBlockSet, self).__init__()
#         self.d = d
#         self.max_r = max_r
#         self.k = k
#         rlist = [1 << x for x in range(15) if (1 << x) <= max_r]
#         self.blocks = nn.Sequential(*[ResBlock(d, r, k, casual) for r in rlist])
#
#     def forward(self, input):
#         x = input
#         x = self.blocks(x)
#         return x
#
#
# class BytenetEncoder(nn.Module):
#     """
#         d = hidden units
#         max_r = maximum dilation rate (paper default: 16)
#         k = masked kernel size (paper default: 3)
#         num_sets = number of residual sets (paper default: 6. 5x6 = 30 ResBlocks)
#         a = relative length of output sequence
#         b = output sequence length intercept
#     """
#     def __init__(self, input_shape, d=128, max_r=16, k=3, num_sets=6):
#         super(BytenetEncoder, self).__init__()
#         self.pad = not (input_shape[0] % 2 == 0)
#         self.d = d
#         self.max_r = max_r
#         self.k = k
#         self.num_sets = num_sets
#         self.pad_in = nn.ConstantPad1d((0, 1), 0.)
#         self.conv_in = nn.Conv1d(input_shape[2], 2*d, 1)
#         self.sets = nn.Sequential()
#         for i in range(num_sets):
#             self.sets.add_module("set_{}".format(i+1), ResBlockSet(d, max_r, k))
#         self.conv_out = nn.Conv1d(2*d, 2*d, 1)
#         self.project = nn.Sequential(
#             nn.Flatten(),
#
#             nn.LazyLinear(32),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, input):
#         x = torch.squeeze(input, len(input.shape) - 2)
#         if self.pad:
#             x = nn.functional.pad(x, (0, 0, 0, 1), "constant", 0)
#         x = torch.transpose(x, len(x.shape) - 1, len(x.shape) - 2)
#         x = self.conv_in(x)
#         x = self.sets(x)
#         x = self.conv_out(x)
#         x = self.project(x)
#         return x