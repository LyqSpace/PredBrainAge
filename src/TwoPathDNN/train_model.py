import os
from optparse import OptionParser

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from src.Database import Database
from torch.autograd import Variable

from src.TwoPathDNN.Net import Net
from src.Logger import Logger


def get_user_params():

    try:
        opts = OptionParser()
        opts.add_option('--retrain',
                        action='store_true',
                        dest='retrain',
                        default=False,
                        help='Retrain the net.')
        opts.add_option('--resample',
                        action='store_true',
                        dest='resample',
                        default=False,
                        help='Resample the data.')
        opts.add_option('--st_id',
                        dest='st_id',
                        type=int,
                        default=0,
                        help='The beginning index of the training pair list in the training process.')
        opts.add_option('--st_lr',
                        dest='st_lr',
                        type=float,
                        default=1e-4,
                        help='The begining learning rate in the training process.')

        options, args = opts.parse_args()
        retrain = options.retrain
        resample = options.resample
        st_id = options.st_id
        st_lr = options.st_lr

        err_messages = []
        check_opts = True
        if st_id < 0:
            err_messages.append('st_id must be a non-negative integer.')
            check_opts = False

        if st_lr <= 0 or st_lr >=1:
            err_messages.append('st_lr must be a float in (0,1).')
            check_opts = False

        if check_opts:
            user_params = {
                'retrain': retrain,
                'resample': resample,
                'st_id': st_id,
                'st_lr': st_lr
            }
            return user_params
        else:
            for err_message in err_messages:
                print(err_message)
            opts.print_help()
            return None

    except Exception as ex:
        print('Exception : %s' % str(ex))
        return None


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


def train_model(net, database, st_id, st_lr):

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=st_lr, alpha=0.9)
    # optimizer = optim.SGD(net.parameters(), lr=st_lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)
    output_step = 10

    logger = Logger('train', 'train.log')

    for epoch in range(100):

        database.set_training_pair_index()
        if epoch == 0:
            database.set_training_pair_index(st_id)

        running_loss = 0
        total_loss = 0
        data_count = 0

        while database.has_training_pair_next():

            data_count += 1

            img_name1, img_tensor1, img_name2, img_tensor2, age_diff_tensor = database.load_training_pair_next()

            img_tensor1 = Variable(img_tensor1.unsqueeze(0).unsqueeze(0).float().cuda())
            img_tensor2 = Variable(img_tensor2.unsqueeze(0).unsqueeze(0).float().cuda())
            age_diff_tensor = Variable(age_diff_tensor.float().cuda())

            output = net(img_tensor1, img_tensor2)
            # print('output: ', output)
            # print('target: ', age_tensor)
            output_float = output.data.cpu().numpy()[0][0]
            age_diff_float = age_diff_tensor.data.cpu().numpy()[0]
            print(database.get_training_pair_index(), img_name1, img_name2, age_diff_float, output_float)

            optimizer.zero_grad()
            loss = criterion(output, age_diff_tensor)
            loss.backward()
            scheduler.step()
            optimizer.step()

            running_loss += loss.data[0]
            total_loss += loss.data[0]

            # if loss.data[0] > 100:
            #     print('Odd res: ', loss.data[0])

            if data_count % output_step == 0:
                message = 'Epoch: %d, Pair id: %d, Data size: %d, Avg loss: %.3f, Last loss: %.3f' % \
                          (epoch, database.get_training_pair_index(), data_count,
                           total_loss / data_count, running_loss / output_step)
                print(message)
                logger.log(message)
                running_loss = 0

            if data_count % output_step+1 == output_step - 1:
                torch.save(net, 'net.pkl')

            # if data_size == 50:
            #     break

        torch.save(net, 'net.pkl')


def main(retrain, resample, st_id, st_lr):

    test_mode = False
    print('Load database. Test mode: %s, Resample: %s' % (test_mode, resample))
    database = Database()
    database.load_database('data/', 'IXI-T1', shape=(128, 128, 75), test_mode=test_mode, resample=resample)

    if retrain is False and os.path.exists(r'net.pkl'):
        print('Construct net. Load from pkl file.')
        net = torch.load('net.pkl')
    else:
        print('Construct net. Create a new network.')
        net = Net()
        init_net_params(net)

    net.float()
    net.train()
    net.cuda()

    print('Start training.')
    train_model(net, database, st_id, st_lr)

if __name__ == '__main__':
    # cudnn.enabled = False

    user_params = get_user_params()
    if user_params is not None:
        main(retrain=user_params['retrain'],
             resample=user_params['resample'],
             st_id=user_params['st_id'],
             st_lr=user_params['st_lr'])
    else:
        raise Exception('User params are wrong.')