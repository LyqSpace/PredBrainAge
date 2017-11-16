import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        conv1_layers = 32
        conv2_layers = conv1_layers * 2
        conv3_layers = conv2_layers * 2
        conv4_layers = conv3_layers * 2
        self.conv1 = nn.Conv3d(1, conv1_layers, (3, 3, 4))
        self.conv2 = nn.Conv3d(conv1_layers, conv2_layers, (3, 3, 3))
        self.conv3 = nn.Conv3d(conv2_layers, conv3_layers, (3, 3, 4))
        self.conv4 = nn.Conv3d(conv3_layers, conv4_layers, (4, 4, 4))

        self.fc1 = nn.Linear(conv4_layers * 3 * 3 * 2, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, (3, 3, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, (2, 2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool3d(x, (2, 2, 2))
        x = F.relu(self.conv4(x))
        x = F.max_pool3d(x, (2, 2, 2))
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
    import time
    time.sleep(1e4)
