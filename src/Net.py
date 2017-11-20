import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # net 1, MAE: 6.6
        #
        conv1_layers = 8
        conv2_layers = conv1_layers * 2
        conv3_layers = conv2_layers * 2

        self._fc_nums = conv3_layers * 4 * 4 * 5

        self.convs1 = nn.Sequential (
            nn.Conv3d(1, conv1_layers, (4, 4, 4)),
            nn.ReLU(inplace=True),
            nn.Conv3d(conv1_layers, conv1_layers, (3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((3, 3, 2))
        )

        self.convs2 = nn.Sequential(
            nn.Conv3d(conv1_layers, conv2_layers, (4, 4, 4)),
            nn.ReLU(inplace=True),
            nn.Conv3d(conv2_layers, conv2_layers, (3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((3, 3, 2)),
        )

        self.convs3 = nn.Sequential(
            nn.Conv3d(conv2_layers, conv3_layers, (3, 3, 4)),
            nn.ReLU(inplace=True),
            nn.Conv3d(conv3_layers, conv3_layers, (3, 3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2))
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self._fc_nums, 512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True)
        )

        self.units = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )

        # net2
        #
        # conv1_layers = 8
        # conv2_layers = conv1_layers * 2
        # conv3_layers = conv2_layers * 2
        # conv4_layers = conv3_layers * 2
        #
        # self._fc_nums = conv4_layers * 3 * 3 * 2
        #
        # self.convs = nn.Sequential (
        #     nn.Conv3d(1, conv1_layers, (3, 3, 3)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(conv1_layers, conv1_layers, (3, 3, 3)),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d((2, 2, 2)),
        #     # nn.Dropout(0.2, inplace=True),
        #
        #     nn.Conv3d(conv1_layers, conv2_layers, (3, 3, 3)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(conv2_layers, conv2_layers, (3, 3, 4)),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d((2, 2, 2)),
        #     # nn.Dropout(0.2, inplace=True),
        #
        #     nn.Conv3d(conv2_layers, conv3_layers, (3, 3, 3)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(conv3_layers, conv3_layers, (3, 3, 3)),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d((3, 3, 2)),
        #     # nn.Dropout(0.2, inplace=True),
        #
        #     nn.Conv3d(conv3_layers, conv4_layers, (3, 3, 3)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(conv4_layers, conv4_layers, (4, 4, 4)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(conv4_layers, conv4_layers, (3, 3, 3)),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d((4, 4, 4)),
        #     # nn.Dropout(0.2, inplace=True),
        #
        # )
        #
        # self.fcs = nn.Sequential(
        #     nn.Linear(self._fc_nums, 1024),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.2, inplace=True),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.2, inplace=True)
        # )
        #
        # self.units = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.2, inplace=True),
        #     nn.Linear(1024, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 1),
        # )

    def forward(self, x1, x2):
        x1 = self.convs1(x1)
        x1 = self.convs2(x1)
        x1 = F.dropout(self.convs3(x1), p=0.2, training=self.training)
        x1 = x1.view(1, self._fc_nums)
        x1 = F.dropout(self.fc1(x1), p=0.2, training=self.training)
        x1 = self.fc2(x1)

        x2 = self.convs1(x2)
        x2 = self.convs2(x2)
        x2 = F.dropout(self.convs3(x2), p=0.2, training=self.training)
        x2 = x2.view(1, self._fc_nums)
        x2 = F.dropout(self.fc1(x2), p=0.2, training=self.training)
        x2 = self.fc2(x2)

        # net1
        # x = torch.cat((x1, x2), 1)

        # net2
        diff_x = x1 - x2
        mul_x = torch.mul(x1, x2)
        x = torch.cat((x1, x2, diff_x, mul_x), 1)
        # unit_x = torch.cat((x1, x2, x1, x2), 1)

        x = self.units(x)

        return x


if __name__ == '__main__':
    # import torch.backends.cudnn as cudnn
    # cudnn.enabled = False
    net = Net()
    print(net)
    net.cuda()
    input = Variable(torch.randn(1, 1, 128, 128, 75).cuda())
    output = net(input, input)
    print(output)
