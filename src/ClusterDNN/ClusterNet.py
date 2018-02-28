import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
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


class ClusterNet(nn.Module):

    def __init__(self, block, layers):

        self.planes = 16
        super(ClusterNet, self).__init__()
        self.conv = nn.Conv3d(1, self.planes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn = nn.BatchNorm3d(self.planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AvgPool3d(kernel_size=(6,7,6))

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

        return x

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True`` then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :func:`state_dict()` function.
        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            strict (bool): Strictly enforce that the keys in :attr:`state_dict`
                match the keys returned by this module's `:func:`state_dict()`
                function.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


def create_cluster_net(**kwargs):

    cluster_net = ClusterNet(BasicBlock, [2, 2, 2])
    return cluster_net


if __name__ == '__main__':

    # official_net = resnet18()
    # print(official_net)
    #
    # print('\n\nend\n\n')

    cluster_net = create_cluster_net()
    print(cluster_net)
    # resnet.cuda()
    # input = Variable(torch.randn(1, 1, 21, 27, 21).cuda())
    input = Variable(torch.randn(1, 1, 91, 109, 91))
    output = cluster_net(input)
    print(output)