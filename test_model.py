import os
from optparse import OptionParser

from src.ClusterDNN.BaselineModel import BaselineModel
from src.ClusterDNN.ClusterModel import ClusterModel


def get_user_params():

    try:
        opts = OptionParser()
        opts.add_option('--validate',
                        action='store_true',
                        dest='validate',
                        default=False,
                        help='Validate the model.')
        opts.add_option('--model_epoch',
                        action='store',
                        type='int',
                        default=0,
                        dest='model_epoch',
                        help='Input the model epoch to load the model to validate.')
        opts.add_option('--baseline',
                        action='store_true',
                        dest='baseline',
                        default=False,
                        help='Train the baseline model.')
        opts.add_option('--cpu',
                        action='store_true',
                        dest='cpu',
                        default=False,
                        help='Test the model by cpu.')

        options, args = opts.parse_args()
        validate = options.validate
        model_epoch = options.model_epoch
        baseline = options.baseline
        use_cpu = options.cpu

        err_messages = []
        check_opts = True

        if check_opts:
            user_params = {
                'validate': validate,
                'model_epoch': model_epoch,
                'baseline': baseline,
                'use_cpu': use_cpu
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


def main(validate, use_cpu, baseline, model_epoch):

    data_path = 'data/'

    if baseline:
        model = BaselineModel()
    else:
        model = ClusterModel()

    if validate:
        model.test(data_path, model_epoch=model_epoch, use_cpu=use_cpu, mode='validation')
    else:
        model.test(data_path, model_epoch=model_epoch, use_cpu=use_cpu, mode='test')


if __name__ == '__main__':

    user_params = get_user_params()

    if user_params is not None:
        main(validate=user_params['validate'],
             model_epoch=user_params['model_epoch'],
             baseline=user_params['baseline'],
             use_cpu=user_params['use_cpu'])
    else:
        raise Exception('User params are wrong.')