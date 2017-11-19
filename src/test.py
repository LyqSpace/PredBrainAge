import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn

from src.Database import Database


def test_model(net, database):

    criterion = nn.MSELoss()
    database.reset_test_index()
    running_loss = 0
    data_size = 0

    while database.has_test_next():

        img_name, img_tensor, age_tensor = database.load_test_data_next()
        # print(database.get_test_index(), img_name)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).float()
        img_tensor = Variable(img_tensor.cuda())
        age_tensor = age_tensor.float()
        age_tensor = Variable(age_tensor.cuda())

        output = net(img_tensor)
        # print('output: ', output)
        # print('target: ', age_tensor)
        loss = criterion(output, age_tensor)

        running_loss += loss.data[0] ** 0.5
        data_size += 1

        if data_size % 11 == 10:
            print('Test size: %d, MAE: %.3f' % (data_size, running_loss / data_size))

    running_loss /= data_size
    print('Test size: %d, MAE: %.3f' % (data_size, running_loss))


def main(pre_train=False):
    print('Load database.')
    database = Database()
    database.load_database('IXI-T1', shape=(128, 128, 75), resample=False)

    if pre_train and os.path.exists(r'net.pkl'):
        print('Construct net. Load from pkl file.')
        net = torch.load('net.pkl')
    else:
        raise Exception('Test the net need net.kpl file.')

    net.cuda()

    print('Start testing.')
    test_model(net, database)


if __name__ == '__main__':
    cudnn.enabled = False
    main(pre_train=True)