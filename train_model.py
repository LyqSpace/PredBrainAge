from optparse import OptionParser
from src.SparseDict.Database import Database
from src.SparseDict.SparseDict import SparseDict


def get_user_params():

    try:
        opts = OptionParser()
        opts.add_option('--cpu',
                        action='store_true',
                        dest='cpu',
                        default=False,
                        help='Train the model by cpu.')
        opts.add_option('--st_epoch',
                        action='store',
                        type='int',
                        default=0,
                        dest='st_epoch',
                        help='Input the start epoch and load net from the file.')
        opts.add_option('--resample',
                        action='store_true',
                        dest='resample',
                        default=False,
                        help='Resample the dataset.')

        options, args = opts.parse_args()
        use_cpu = options.cpu
        st_epoch = options.st_epoch
        resample = options.resample

        err_messages = []
        check_opts = True

        if check_opts:
            user_params = {
                'st_epoch': st_epoch,
                'use_cpu': use_cpu,
                'resample': resample
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


def main(use_cpu, st_epoch, resample):

    data_path = 'data/'
    expt_path = 'expt/'
    dataset_name = 'IXI-T1'

    database = Database()
    database.integrate_data(data_path, dataset_name, resample)

    sparse_dict = SparseDict(data_path, expt_path, n_components=128, mode='training')
    sparse_dict.fit()

    # model.train(st_epoch=st_epoch)


if __name__ == '__main__':

    user_params = get_user_params()

    if user_params is not None:
        main(use_cpu=user_params['use_cpu'],
             st_epoch=user_params['st_epoch'],
             resample=user_params['resample'])
    else:
        raise Exception('User params are wrong.')