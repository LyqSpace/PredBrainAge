import os
from optparse import OptionParser

from src.Logger import Logger
from src.IL.InductiveLearning import InductiveLearning


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


def main(resample):

    inductive_model = InductiveLearning()
    inductive_model.train('data/', 'IXI-T1', resample=resample)


if __name__ == '__main__':

    user_params = get_user_params()

    if user_params is not None:
        main(resample=user_params['resample'])
    else:
        raise Exception('User params are wrong.')