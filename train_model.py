import os
from optparse import OptionParser

from src.Logger import Logger
from src.ClusterDNN.ClusterModel import ClusterModel
from src.ClusterDNN.BaselineModel import BaselineModel


def get_user_params():

    try:
        opts = OptionParser()
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
        opts.add_option('--pretrain_model',
                        action='store',
                        type='int',
                        default=0,
                        dest='pretrain_model',
                        help='Input the epoch of baseline model as the pretrain net from the file.')

        options, args = opts.parse_args()
        use_cpu = options.cpu
        baseline = options.baseline
        st_epoch = options.st_epoch
        pretrain_model = options.pretrain_model

        err_messages = []
        check_opts = True

        if baseline:
            if pretrain_model > 0:
                err = '--baseline and --pretrain_model can not be used at the same time.'
                err_messages.append(err)
        else:
            if pretrain_model > 0 and st_epoch > 0:
                err = '--pretrain_model and --st_epoch can not be used at the same time.'
                err_messages.append(err)
            if pretrain_model > 0 and st_epoch < 0:
                err = '--pretrain_model and --st_epoch can not be negative at the same time.'
                err_messages.append(err)

        if check_opts:
            user_params = {
                'st_epoch': st_epoch,
                'use_cpu': use_cpu,
                'baseline': baseline,
                'pretrain_model': pretrain_model
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


def main(use_cpu, baseline, st_epoch, pretrain_model):

    data_path = 'data/'
    dataset_name = 'IXI-T1'

    if baseline:
        model = BaselineModel(data_path, mode='training', use_cpu=use_cpu)
    else:
        model = ClusterModel(data_path, mode='training', use_cpu=use_cpu)
        if st_epoch == 0:
            model.load_pretrain(pretrain_model)

    model.train(st_epoch=st_epoch)


if __name__ == '__main__':

    user_params = get_user_params()

    if user_params is not None:
        main(use_cpu=user_params['use_cpu'],
             baseline = user_params['baseline'],
             st_epoch=user_params['st_epoch'],
             pretrain_model = user_params['pretrain_model'])
    else:
        raise Exception('User params are wrong.')