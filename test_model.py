import os
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from src.Database import Database


def test_model(net, database):

    database.set_test_index()
    test_data_count = 0
    total_loss = 0
    training_data_size = 10

    while database.has_test_next():

        img_name, img_tensor, age_tensor = database.load_test_data_next()
        target_age = age_tensor.numpy()[0]
        print('img_name: %s, target_age: %.3f' % (img_name, target_age))
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).float()
        img_tensor = Variable(img_tensor.cuda())

        database.random_training_data(training_data_size)
        training_data_count = 0
        age_sum = 0

        while database.has_training_data_next():

            training_img_name, training_img_tensor, training_age_tensor = database.load_training_data_next()
            training_age = training_age_tensor.numpy()[0]
            training_img_tensor = training_img_tensor.unsqueeze(0).unsqueeze(0).float()
            training_img_tensor = Variable(training_img_tensor.cuda())
            output = net(img_tensor, training_img_tensor)
            # print('output: ', output)
            # print('target: ', age_tensor)
            output = output.data.cpu().numpy()[0][0]
            output = output + training_age
            age_sum += output
            training_data_count += 1

            print('    Count: %d, Train id: %s, Target Age: %.3f, Mean Age: %.3f' % (training_data_count,
                                                                                     training_img_name,
                                                                                     target_age,
                                                                                     age_sum / training_data_count))

        test_data_count += 1
        age_sum /= training_data_count
        total_loss += abs(age_sum - target_age)

    total_loss /= test_data_count
    print('Test Size: %d, MAE: %.3f' % (test_data_count, total_loss))


def main():
    print('Load database.')
    database = Database()
    database.load_database('data/', 'IXI-T1', shape=(128, 128, 75), test=True, resample=False)

    if os.path.exists(r'net.pkl.backup'):
        print('Construct net. Load from pkl.backup file.')
        net = torch.load('net.pkl.backup')
    else:
        raise Exception('Test the net need net.kpl.backup file.')

    net.cuda()

    print('Start testing.')
    test_model(net, database)


if __name__ == '__main__':
    # cudnn.enabled = False
    main()