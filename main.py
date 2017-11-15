import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from src.Net import Net
from src.Database import Database


def train_model(net, database):

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):

        database.reset_training_index()

        while database.has_training_next():

            img_name, img_tensor, age_tensor = database.load_training_data_next()
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).float()
            img_tensor = Variable(img_tensor.cuda())
            age_tensor = age_tensor.float()
            age_tensor = Variable(age_tensor.cuda())

            output = net(img_tensor)
            print(output)

            optimizer.zero_grad()
            loss = criterion(output, age_tensor)
            loss.backward()
            optimizer.step()

            if database.get_training_index() % 1 == 0:
                print('Epoch: %d, Data: %d, Loss: %.3f' % (epoch, database.get_training_index(), loss.data[0]))

            break


def main():
    print('Load database.')
    database = Database()
    database.load_database('IXI-T1')

    print('Construct net.')
    net = Net()
    net.cuda()

    print('Start training.')
    train_model(net, database)

if __name__ == '__main__':
    cudnn.enabled = False
    main()
