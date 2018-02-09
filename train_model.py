import os
from optparse import OptionParser

from src.Logger import Logger
from src.DL.DivideLearning import DivideLearning


def get_user_params():

    try:
        opts = OptionParser()
        opts.add_option('--resample',
                        action='store_true',
                        dest='resample',
                        default=False,
                        help='Resample the data.')
        opts.add_option('--retrain',
                        action='store_true',
                        dest='retrain',
                        default=False,
                        help='Retrain the net.')
        opts.add_option('--cpu',
                        action='store_true',
                        dest='cpu',
                        default=False,
                        help='Train the model by cpu.')
        opts.add_option('--divide',
                        action='store_true',
                        dest='divide',
                        default=False,
                        help='Divide process.')
        opts.add_option('--induce',
                        action='store_true',
                        dest='induce',
                        default=False,
                        help='Induce process.')
        opts.add_option('--st_epoch',
                        action='store',
                        type='int',
                        default=0,
                        dest='st_epoch',
                        help='Input the start epoch and load net from the file.')

        options, args = opts.parse_args()
        resample = options.resample
        retrain = options.retrain
        use_cpu = options.cpu
        st_epoch = options.st_epoch
        divide = options.divide
        induce = options.induce

        err_messages = []
        check_opts = True

        if check_opts:
            user_params = {
                'resample': resample,
                'retrain': retrain,
                'st_epoch': st_epoch,
                'use_cpu': use_cpu,
                'divide': divide,
                'induce': induce
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


def main(resample, retrain, use_cpu, divide, induce, st_epoch):

    data_path = 'data/'
    dataset_name = 'IXI-T1'

    divide_model = DivideLearning()

    if divide:
        divide_model.divide(data_path, dataset_name, resample=resample)

    if induce:
        divide_model.induce(data_path, dataset_name, retrain=retrain, use_cpu=use_cpu, st_epoch=st_epoch)


if __name__ == '__main__':

    user_params = get_user_params()

    if user_params is not None:
        main(resample=user_params['resample'],
             retrain=user_params['retrain'],
             use_cpu=user_params['use_cpu'],
             divide=user_params['divide'],
             st_epoch=user_params['st_epoch'],
             induce=user_params['induce'])
    else:
        raise Exception('User params are wrong.')