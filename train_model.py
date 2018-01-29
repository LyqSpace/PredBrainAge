import os
from optparse import OptionParser

from src.Logger import Logger
from src.IL.InductiveLearning import InductiveLearning


def get_user_params():

    try:
        opts = OptionParser()
        opts.add_option('--resample',
                        action='store_true',
                        dest='resample',
                        default=False,
                        help='Resample the data.')
        opts.add_option('--st_group',
                        dest='st_group',
                        type=int,
                        default=0,
                        help='The beginning group age in the training process.')
        opts.add_option('--divide',
                        action='store_true',
                        dest='divide',
                        default=False,
                        help='Divide process.')
        opts.add_option('--induce',
                        action="store_true",
                        dest='induce',
                        default=False,
                        help='Induce process.')

        options, args = opts.parse_args()
        resample = options.resample
        st_group = options.st_group
        divide = options.divide
        induce = options.induce

        err_messages = []
        check_opts = True
        if st_group < 0:
            err_messages.append('st_group must be a non-negative integer.')
            check_opts = False

        if check_opts:
            user_params = {
                'resample': resample,
                'st_group': st_group,
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


def main(st_group, resample, divide, induce):

    data_path = 'data/'
    dataset_name = 'IXI-T1'

    inductive_model = InductiveLearning()

    if divide:
        inductive_model.train(data_path, dataset_name, st_group=st_group, resample=resample)

    if induce:
        inductive_model.induce(data_path, dataset_name)


if __name__ == '__main__':

    user_params = get_user_params()

    if user_params is not None:
        main(st_group=user_params['st_group'],
             resample=user_params['resample'],
             divide=user_params['divide'],
             induce=user_params['induce'])
    else:
        raise Exception('User params are wrong.')