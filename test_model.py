import os
from optparse import OptionParser

from src.Logger import Logger
from src.DL.DivideLearning import DivideLearning


def get_user_params():

    try:
        opts = OptionParser()
        opts.add_option('--validate',
                        action='store_true',
                        dest='validate',
                        default=False,
                        help='Validate the model.')

        options, args = opts.parse_args()
        validate = options.validate

        err_messages = []
        check_opts = True

        if check_opts:
            user_params = {
                'validate': validate,
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


def main(validate):

    data_path = 'data/'
    dataset_name = 'IXI-T1'

    divide_model = DivideLearning()

    if validate:
        divide_model.test(data_path, dataset_name, mode='validation')
    else:
        divide_model.test(data_path, dataset_name, mode='test')


if __name__ == '__main__':

    user_params = get_user_params()

    if user_params is not None:
        main(validate=user_params['validate'])
    else:
        raise Exception('User params are wrong.')