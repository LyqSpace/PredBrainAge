import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class BaselineNet(nn.Module):

    def __init__(self, block, layers):

        self.planes = 16
        super(BaselineNet, self).__init__()
        self.conv = nn.Conv3d(1, self.planes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn = nn.BatchNorm3d(self.planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AvgPool3d(kernel_size=(6,7,6))
        self.fc = nn.Linear(64 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv3d(self.planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.planes, planes, stride, down_sample))
        self.planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def create_baseline_net(**kwargs):

    model = BaselineNet(BasicBlock, [2, 2, 2])
    return model


if __name__ == '__main__':

    # official_net = resnet18()
    # print(official_net)
    #
    # print('\n\nend\n\n')

    baseline_net = create_baseline_net()
    print(baseline_net)
    # resnet.cuda()
    # input = Variable(torch.randn(1, 1, 21, 27, 21).cuda())
    input = Variable(torch.randn(1, 1, 91, 109, 91))
    output = baseline_net(input)
    print(output)