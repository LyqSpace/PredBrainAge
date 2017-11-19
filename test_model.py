import os
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from src.Database import Database


def test_model(net, database):

    database.set_test_index()
    test_data_count = 0
    total_loss = 0
    training_data_size = 20

    while database.has_test_next():

        img_name, img_tensor, age_tensor = database.load_test_data_next()
        target_age = age_tensor.numpy()[0]
        print('Test id: %d, Img name: %s, Target age: %.3f' % (database.get_test_index(), img_name, target_age))
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).float()
        img_tensor = Variable(img_tensor.cuda())

        database.select_training_data(training_data_size, target_age)
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
            age_diff = output.data.cpu().numpy()[0][0]
            pred_age = age_diff + training_age
            if pred_age < 10 or pred_age > 95:
                print('Odd res !!')
            age_sum += pred_age
            training_data_count += 1

            print('    Count: %d, Name: %s, Age: %.3f, Diff %.3f, Pred Age: %.3f, Mean Res: %.3f' % (training_data_count,
                                                                                                     training_img_name,
                                                                                                     training_age,
                                                                                                     age_diff,
                                                                                                     pred_age,
                                                                                                     age_sum / training_data_count))

        test_data_count += 1
        age_sum /= training_data_count
        total_loss += abs(age_sum - target_age)

    total_loss /= test_data_count
    print('Test size: %d, MAE: %.3f' % (test_data_count, total_loss))


def main():
    test_mode = True
    resample = False
    print('Load database. Test mode %s, Resample %s' % (test_mode, resample))
    database = Database()
    database.load_database('data/', 'IXI-T1', shape=(128, 128, 75), test_mode=test_mode, resample=resample)

    if os.path.exists(r'net.pkl'):
        print('Construct net. Load from pkl file.')
        net = torch.load('net.pkl')
    else:
        raise Exception('Test the net need net.kpl file.')

    net.cuda()

    print('Start testing.')
    test_model(net, database)


if __name__ == '__main__':
    # cudnn.enabled = False
    main()