import torch.nn as nn
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):

        first_planes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, first_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm3d(first_planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, first_planes, layers[0])
        self.layer2 = self._make_layer(block, first_planes*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, first_planes*4, layers[2], stride=2)
        self.avg_pool = nn.AvgPool3d(7)
        self.fc = nn.Linear(first_planes*4 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.first_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv3d(self.first_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.first_planes, planes, stride, down_sample))
        self.first_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.first_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(**kwargs):

    model = ResNet(BasicBlock, [2, 2, 2], **kwargs)
    return model