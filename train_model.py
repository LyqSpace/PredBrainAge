import os
import torch
from optparse import OptionParser
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

from src.Net import Net
from src.Database import Database


def get_user_params():

    try:
        opts = OptionParser()
        opts.add_option('--st_index',
                        dest='st_index',
                        type=int,
                        default=0,
                        help='The beginning index of the training pair list in the training process.')
        opts.add_option('--st_lr',
                        dest='st_lr',
                        type=float,
                        default=1e-4,
                        help='The begining learning rate in the training process.')

        options, args = opts.parse_args()
        st_index = options.st_index
        st_lr = options.st_lr

        err_messages = []
        check_opts = True
        if st_index < 0:
            err_messages.append('st_index must be a non-negative integer.')
            check_opts = False

        if st_lr <= 0 or st_lr >=1:
            err_messages.append('st_lr must be a float in (0,1).')
            check_opts = False

        if check_opts:
            user_params = {
                'st_index': st_index,
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


def train_model(net, database, st_index, st_lr):

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=st_lr, alpha=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    output_step = 10

    for epoch in range(100):

        database.set_training_pair_index()
        if epoch == 0:
            database.set_training_pair_index(st_index)

        running_loss = 0
        total_loss = 0
        data_count = 0

        while database.has_training_pair_next():

            data_count += 1

            img_name1, img_tensor1, img_name2, img_tensor2, age_diff_tensor = database.load_training_pair_next()
            print(database.get_training_pair_index(), img_name1, img_name2)

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
            total_loss += loss.data[0]

            if loss.data[0] > 100:
                print(loss.data[0])

            if data_count % output_step == output_step - 1:
                print('Epoch: %d, Data size: %d, Total loss: %.3f, Last loss: %.3f' % (epoch, data_count,
                                                                                  total_loss / data_count,
                                                                                  running_loss / (output_step+1)))
                running_loss = 0

            if data_count % output_step+1 == output_step - 1:
                torch.save(net, 'net.pkl')

            # if data_size == 50:
            #     break

        torch.save(net, 'net.pkl')

        running_loss /= data_count
        print('=== Epoch: %d, Data size: %d, Average loss: %.3f' % (epoch, data_count, running_loss))


def main(pre_train, st_index, st_lr):
    print(st_lr)
    test_mode = False
    resample = False
    print('Load database. Test mode %s, Resample %s' % (test_mode, resample))
    database = Database()
    database.load_database('data/', 'IXI-T1', shape=(128, 128, 75), test_mode=test_mode, resample=resample)

    if pre_train and os.path.exists(r'net.pkl'):
        print('Construct net. Load from pkl file.')
        net = torch.load('net.pkl')
    else:
        print('Construct net. Create a new network.')
        net = Net()
        # init_net_params(net)
    net.cuda()

    print('Start training.')
    train_model(net, database, st_index, st_lr)

if __name__ == '__main__':
    # cudnn.enabled = False

    user_params = get_user_params()
    if user_params is not None:
        main(pre_train=True,
             st_index=user_params['st_index'],
             st_lr=user_params['st_lr'])
    else:
        raise Exception('User params is wrong.')