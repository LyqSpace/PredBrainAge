import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 5, (6, 6, 6))
        self.conv2 = nn.Conv3d(5, 10, (5, 5, 6))
        self.conv3 = nn.Conv3d(10, 15, (6, 6, 6))
        self.conv4 = nn.Conv3d(15, 20, (5, 5, 4))

        self.fc1 = nn.Linear(20 * 6 * 6 * 2, 150)
        self.fc2 = nn.Linear(150, 15)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), (2, 2, 2))
        x = F.max_pool3d(F.relu(self.conv2(x)), (3, 3, 3))
        x = F.max_pool3d(F.relu(self.conv3(x)), (2, 2, 2))
        x = F.max_pool3d(F.relu(self.conv4(x)), (2, 2, 2))
        x = x.view(1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    net = Net()
    input = Variable(torch.randn(1, 1, 256, 256, 150))
    output = net(input)
    print(output)
