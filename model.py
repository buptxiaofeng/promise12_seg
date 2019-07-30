import torch
import torch.nn as nn
import torch.nn.functional as F

class InputTransition(nn.Module):
    def __init__(self, out_channels):
        super(InputTransition, self).__init__()
        self.conv = nn.Conv3d(1, 16, kernel_size = 5, padding = 2)
        self.bn = nn.BatchNorm3d(16)
        self.elu = nn.ELU(inplace = True)

    def forward(self, x):
        out = self.bn(self.conv(x))
        x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 1)
        out = self.elu(torch.add(out, x16))

        return out

class LUConv(nn.Module):
    def __init__(self, n_channels):
        super(LUConv, self).__init__()
        self.elu = nn.ELU(inplace = True)
        self.conv = nn.Conv3d(n_channels, n_channels, kernel_size = 5, padding = 2)
        self.bn = nn.BatchNorm3d(n_channels)

    def forward(self, x):
        out = self.elu(self.bn(self.conv(x)))

        return out

def _make_conv(n_channels, depth):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(n_channels))

    return nn.Sequential(*layers)

class DownTransition(nn.Module):
    def __init__(self, in_channels, n_channels):
        super(DownTransition, self).__init__()
        out_channels = 2 * in_channels
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size = 2, stride = 2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout3d()
        self.ops = _make_conv(out_channels, n_channels)

    def forward(self, x):
        down = self.elu(self.bn(self.down_conv(x)))
        out = self.dropout(down)
        out = self.ops(out)
        out = self.elu(torch.add(out, down))

        return out

class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size = 2, stride = 2)
        self.bn = nn.BatchNorm3d(out_channels // 2)
        self.dropout = nn.Dropout3d()
        self.elu = nn.ELU()
        self.ops = _make_conv(out_channels, n_conv)

    def forward(self, x1, x2):
        out1 = self.dropout(x1)
        out1 = self.elu(self.bn(self.up_conv(out1)))
        out2 = self.dropout(x2)
        out_cat = torch.cat((out1, out2), 1)
        out = self.ops(out_cat)
        out = self.elu(torch.add(out, out_cat))

        return out

class OutputTransition(nn.Module):
    def __init__(self, in_channels):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 2, kernel_size = 5, padding = 2)
        self.bn = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size = 1)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        out = self.elu(self.bn(self.conv1(x)))
        out = self.conv2(out)
        #b:batch_size, c:channels, z:depth, y:height, w:width
        b, c, z, y, x = out.shape
        out = out.view(b, c, -1)
        res = self.softmax(out)

        return res

class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        self.input_tr = InputTransition(16)
        self.down_tr32 = DownTransition(16, 1)
        self.down_tr64 = DownTransition(32, 2)
        self.down_tr128 = DownTransition(64, 3)
        self.down_tr256 = DownTransition(128, 2)
        self.up_tr256 = UpTransition(256, 256, 2)
        self.up_tr128 = UpTransition(256, 128, 2)
        self.up_tr64 = UpTransition(128, 64, 1)
        self.up_tr32 = UpTransition(64, 32, 1)
        self.out_tr = OutputTransition(32)

    def forward(self, x):
        out16 = self.input_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)

        return out
