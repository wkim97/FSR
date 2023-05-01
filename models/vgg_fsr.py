import torch
import torch.nn as nn
from models.gumbel_sigmoid import GumbelSigmoid


__all__ = [
    'vgg16_FSR',
]


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
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, feat, is_eval=False):
        mask = self.sep_net(feat)

        mask = mask.reshape(mask.shape[0], 1, -1)
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


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True, tau=0.1, image_size=(32, 32)):
        super(VGG, self).__init__()
        self.image_size = image_size

        self.block0, self.pool0, self.block1, self.pool1, self.block2, self.pool2, self.block3, self.pool3, self.block4, self.pool4 = features
        self.tau = tau

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

        self.separation = Separation(size=(512, int(self.image_size[0] / 8), int(self.image_size[1] / 8)), tau=self.tau)
        self.recalibration = Recalibration(size=(512, int(self.image_size[0] / 8), int(self.image_size[1] / 8)))
        self.aux = nn.Sequential(nn.Linear(512, num_classes))

        if init_weights:
            self._initialize_weights()

    def forward(self, x, is_eval=False):
        r_outputs = []
        nr_outputs = []
        rec_outputs = []

        x = self.block0(x)
        x = self.pool0(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)

        r_feat, nr_feat, mask = self.separation(x, is_eval=is_eval)
        r_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1))
        r_outputs.append(r_out)
        nr_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1))
        nr_outputs.append(nr_out)

        rec_feat = self.recalibration(nr_feat, mask)
        rec_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        rec_outputs.append(rec_out)

        x = r_feat + rec_feat

        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x, r_outputs, nr_outputs, rec_outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    features = []
    in_channels = 3
    for v in cfg:
        if v == 'fsr':
            continue
        elif v == 'M':
            features.append(nn.Sequential(*layers))
            layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            features.append(nn.Sequential(*layers))
            layers = []
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return features


cfgs = {
    'vgg16_FSR': ['fsr', 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, tau=0.1, num_classes=10, image_size=(32, 32), **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes=num_classes, image_size=image_size, init_weights=True, tau=tau, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg16_FSR(pretrained=False, progress=True, tau=0.1, num_classes=10, image_size=(32, 32), **kwargs):
    return _vgg('vgg16_FSR', 'vgg16_FSR', True, pretrained, progress, tau=tau, num_classes=num_classes, image_size=image_size, **kwargs)

