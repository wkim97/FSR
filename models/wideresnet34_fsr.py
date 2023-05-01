import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gumbel_sigmoid import GumbelSigmoid


class Separation(torch.nn.Module):
    def __init__(self, size, num_channel=64, tau=0.1):
        super(Separation, self).__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.tau = tau

        self.sep_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, feat, is_eval=False):
        rob_map = self.sep_net(feat)

        mask = rob_map.reshape(rob_map.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)
        mask = GumbelSigmoid(tau=self.tau)(mask, is_eval=is_eval)
        mask = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W)

        r_feat = feat * mask
        nr_feat = feat * (1 - mask)

        return r_feat, nr_feat, mask


class Recalibration(nn.Module):
    def __init__(self, size, num_channel=64):
        super(Recalibration, self).__init__()
        C, H, W = size
        self.rec_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, nr_feat, mask):
        rec_units = self.rec_net(nr_feat)
        rec_units = rec_units * (1 - mask)
        rec_feat = nr_feat + rec_units

        return rec_feat


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, tau=0.1, image_size=(32, 32)):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.image_size = image_size
        self.tau = tau
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.separation = Separation(size=(640, int(self.image_size[0] / 4), int(self.image_size[1] / 4)), tau=self.tau)
        self.recalibration = Recalibration(size=(640, int(self.image_size[0] / 4), int(self.image_size[1] / 4)))
        self.aux = nn.Sequential(nn.Linear(640, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, is_eval=False):
        r_outputs = []
        nr_outputs = []
        rec_outputs = []

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        r_feat, nr_feat, mask = self.separation(out, is_eval=is_eval)
        r_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1))
        r_outputs.append(r_out)
        nr_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1))
        nr_outputs.append(nr_out)

        rec_feat = self.recalibration(nr_feat, mask)
        rec_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        rec_outputs.append(rec_out)

        out = r_feat + rec_feat

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)

        return out, r_outputs, nr_outputs, rec_outputs


def WideResNet34_FSR(num_classes=10, tau=0.1, image_size=(32, 32)):
    return WideResNet(num_classes=num_classes, tau=tau, image_size=image_size)
