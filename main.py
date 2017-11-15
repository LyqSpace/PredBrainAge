import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from src.Net import Net
from src.Database import Database


def train_model(net, database):

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):

        running_loss = 0
        while database.has_next():

            img_name, img_tensor, age_tensor = database.load_data_next()
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).float()
            print(img_tensor.size())
            print(img_tensor.type())
            print(age_tensor.type())
            img_tensor = Variable(img_tensor)
            age_tensor = Variable(age_tensor.float())

            optimizer.zero_grad()

            output = net(img_tensor)
            print(output)
            loss = criterion(output, age_tensor)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if database.get_index() % 1 == 0:
                print('Epoch: %d, Data: %d, Loss: %.3f' % (epoch, database.get_index(), running_loss))
                running_loss = 0


def main():
    print('Load database.')
    database = Database()
    database.load_database('IXI-T1')

    print('Construct net.')
    net = Net()

    print('Start training.')
    train_model(net, database)

if __name__ == '__main__':
    main()
