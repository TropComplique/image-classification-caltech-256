import torch
import torch.nn as nn
import numpy as np
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
}


def get_survival_rate(l, L, p_L=0.5):
    return 1.0 - (l/L)*(1 - p_L)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, survival_rate=1.0, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.survival_rate = survival_rate

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual

        if not self.training or torch.rand(1)[0] < self.survival_rate:
            
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            
            if self.training:
                out /= self.survival_rate

            out += residual
            out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        L = sum(layers)
        layers_cumsum = np.cumsum([0] + layers)
        survival_rates = [
            [get_survival_rate(i + 1, L) for i in range(layers_cumsum[j], layers_cumsum[j + 1])]
            for j in range(0, 4)
        ]

        self.layer1 = self._make_layer(64, layers[0], survival_rates[0])
        self.layer2 = self._make_layer(128, layers[1], survival_rates[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], survival_rates[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], survival_rates[3], stride=2)
        self.avgpool = nn.AvgPool2d(10)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, survival_rates=None, stride=1):
        downsample = None
        if survival_rates is None:
            survival_rates = [1.0]*blocks
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        layers = [BasicBlock(self.inplanes, planes, survival_rates[0], stride, downsample)]
        self.inplanes = planes

        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, survival_rates[i]))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model
