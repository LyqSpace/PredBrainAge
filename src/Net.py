import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 3, (3, 3, 3))
        self.conv2 = nn.Conv3d(3, 6, (5, 5, 3))

        self.fc1 = nn.Linear(6 * 4 * 4 * 3, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, (5, 5, 4))
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, (5, 5, 5))
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
    import torch.backends.cudnn as cudnn
    cudnn.enabled = False
    net = Net()
    net.cuda()
    input = Variable(torch.randn(1, 1, 128, 128, 75).cuda())
    output = net(input)
    print(output)
