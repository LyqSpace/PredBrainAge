from src.IL.Database import Database
import src.utils as utils


class InductiveLearning:

    def __init__(self):
        self._database = Database()
        self._template_data = None
        self._template_am = None

    def load_group(self):
        pass

    def train(self, data_path, dataset_name, resample=True):

        self._database.load_database(data_path, dataset_name, mode='training', resample=resample)

        print('Train model: Inductive Learning. database. Dataset {0}. Resample {1}.'.format(dataset_name, resample))

        while self._database.get_next_group() is not None:

            group_age = self._database.get_group_index()

            print('\tGroup Age: {0}.'.format(group_age))

            _, self._template_data, _ = self._database.get_next_data_from_group()

            utils.show_3Ddata(self._template_data)

            self._template_am = utils.get_attention_map(self._template_data)

            while self._database.has_next_data_from_group():

                data_name, data, age = self._database.get_next_data_from_group()
                self._inductive_learning(data)

        print('Done.')

    def _inductive_learning(self, data):
        pass

    def validate(self):
        self._database.load_database('../../data/', 'IXI-T1', mode='validation')

        pass
