from optparse import OptionParser
import numpy as np

from src.ClusterDNN.Database import Database


def get_user_params():

    try:
        opts = OptionParser()
        opts.add_option('--resample',
                        action='store_true',
                        dest='resample',
                        default=False,
                        help='Resample the data.')

        options, args = opts.parse_args()
        resample = options.resample

        err_messages = []
        check_opts = True

        if check_opts:
            user_params = {
                'resample': resample,
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

    data_path = 'data/'
    dataset_name = 'IXI-T1'

    print('Initialize database.', data_path, dataset_name, 'Resample:', resample)

    database = Database()

    # Training
    database.load_database(data_path, dataset_name, mode='training', resample=resample)

    training_data = []
    training_ages = []

    while database.has_next_data():
        data_name, data, age = database.get_next_data(required_data=True)
        training_data.append(data)
        training_ages.append(age)

    np.save(data_path + 'training_data.npy', np.array(training_data))
    np.save(data_path + 'training_ages.npy', np.array(training_ages))

    # Validation
    database.load_database(data_path, dataset_name, mode='validation')

    validation_data = []
    validation_ages = []

    while database.has_next_data():
        data_name, data, age = database.get_next_data(required_data=True)
        validation_data.append(data)
        validation_ages.append(age)

    np.save(data_path + 'validation_data.npy', np.array(validation_data))
    np.save(data_path + 'validation_ages.npy', np.array(validation_ages))

    # Validation
    database.load_database(data_path, dataset_name, mode='validation')

    validation_data = []
    validation_ages = []

    while database.has_next_data():
        data_name, data, age = database.get_next_data(required_data=True)
        validation_data.append(data)
        validation_ages.append(age)

    np.save(data_path + 'validation_data.npy', np.array(validation_data))
    np.save(data_path + 'validation_ages.npy', np.array(validation_ages))


if __name__ == '__main__':

    user_params = get_user_params()

    if user_params is not None:
        main(resample=user_params['resample'])
    else:
        raise Exception('User params are wrong.')