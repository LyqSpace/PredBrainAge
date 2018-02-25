import os
from optparse import OptionParser

from src.Logger import Logger
from src.ClusterDNN.ClusterModel import ClusterModel
from src.ClusterDNN.BaselineModel import BaselineModel


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
        opts.add_option('--baseline',
                        action='store_true',
                        dest='baseline',
                        default=False,
                        help='Train the baseline model.')
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
        baseline = options.baseline
        st_epoch = options.st_epoch

        err_messages = []
        check_opts = True

        if check_opts:
            user_params = {
                'resample': resample,
                'retrain': retrain,
                'st_epoch': st_epoch,
                'use_cpu': use_cpu,
                'baseline': baseline
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


def main(resample, retrain, use_cpu, baseline, st_epoch):

    data_path = 'data/'
    dataset_name = 'IXI-T1'

    if baseline:
        model = BaselineModel(data_path, dataset_name, resample=resample)
    else:
        model = ClusterModel(data_path, dataset_name, resample=resample)

    model.train(data_path, retrain=retrain, use_cpu=use_cpu, st_epoch=st_epoch)


if __name__ == '__main__':

    user_params = get_user_params()

    if user_params is not None:
        main(resample=user_params['resample'],
             retrain=user_params['retrain'],
             use_cpu=user_params['use_cpu'],
             baseline = user_params['baseline'],
             st_epoch=user_params['st_epoch'])
    else:
        raise Exception('User params are wrong.')