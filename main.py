import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

from src.Net import Net
from src.Database import Database


def init_net_params(net):

    print('Initialize net parameters by xavier_uniform.')

    for m in net.modules():

        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight)
            init.constant(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)

        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            init.constant(m.bias, 0)


def train_model(net, database):

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=1e-5, alpha=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)

    for epoch in range(100):

        database.reset_training_index()
        running_loss = 0
        data_size = 0

        while database.has_training_next():

            img_name1, img_tensor1, img_name2, img_tensor2, age_diff_tensor = database.load_training_data_next()
            print(database.get_training_index(), img_name1, img_name2)
            img_tensor1 = Variable(img_tensor1.unsqueeze(0).unsqueeze(0).float().cuda())
            img_tensor2 = Variable(img_tensor2.unsqueeze(0).unsqueeze(0).float().cuda())
            age_diff_tensor = Variable(age_diff_tensor.float().cuda())

            output = net(img_tensor1, img_tensor2)
            # print('output: ', output)
            # print('target: ', age_tensor)
            optimizer.zero_grad()
            loss = criterion(output, age_diff_tensor)
            loss.backward()
            scheduler.step()
            optimizer.step()

            running_loss += loss.data[0]
            data_size += 1

            if data_size % 11 == 10:
                print('Epoch: %d, Data: %d, Loss: %.3f' % (epoch, database.get_training_index(), running_loss / data_size))

            if data_size % 21 == 20:
                torch.save(net, 'net.pkl')

            # if data_size == 50:
            #     break

        torch.save(net, 'net.pkl')

        running_loss /= data_size
        print('=== Epoch: %d, Datasize: %d, Average Loss: %.3f' % (epoch, data_size, running_loss))


def main(pre_train=False):
    print('Load database.')
    database = Database()
    database.load_database('IXI-T1', shape=(128, 128, 75), resample=False)

    if pre_train and os.path.exists(r'net.pkl'):
        print('Construct net. Load from pkl file.')
        net = torch.load('net.pkl')
    else:
        print('Construct net. Create a new network.')
        net = Net()
        # init_net_params(net)
    net.cuda()

    print('Start training.')
    train_model(net, database)

if __name__ == '__main__':
    # cudnn.enabled = False
    main(pre_train=True)
