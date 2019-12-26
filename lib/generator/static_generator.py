import pandas as pd
import random as rand
from glob import glob
import time
import os


class StaticExchange:
    """A static instrument exchange, in which the price history is loaded from a csv file"""

    def __init__(self, config, **kwargs):
        self.data_frame = pd.DataFrame()
        self.market = config['market']
        self._parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._train_dir = os.path.join(self._parent_dir, 'data', 'dataset', self.market, 'Train_data')
        self._test_dir = os.path.join(self._parent_dir, 'data', 'dataset', self.market, 'Test_data')
        self._current_stock = 'NAN'
        self._file_paths = []
        self.load_file_names()
        self.reset()

    def load_file_names(self):
        self._file_paths = glob(self._train_dir + '/*.csv')

    def load_csv(self, choice=0):
        choice = choice % len(self._file_paths)
        _file_path = self._file_paths[choice]
        self._current_stock = _file_path.split('/')[-1].split('.')[0]
        if os.path.exists(_file_path):
            self.data_frame = pd.read_csv(_file_path)
            if self.market == 'in_mkt':
                self.data_frame.rename(columns={"Date": "date",
                                                "Open": "Open",
                                                "High": "high",
                                                "Low": "low",
                                                "Close": "close"}, inplace=True)
                self.data_frame['date'] = pd.to_datetime(self.data_frame['date'], format='%d-%m-%Y %I:%M:%S %p')
            else:
                self.data_frame['date'] = pd.to_datetime(self.data_frame['date'])
            self.data_frame['date'] = self.data_frame['date'].astype(str)
            self.data_frame = self.data_frame.sort_values(['date'])
            self.data_frame = self.data_frame.set_index('date')
        else:
            print("File not Found!")

    def reset(self, choice=0):
        self.data_frame = pd.DataFrame()
        self.load_csv(choice)

        return self._current_stock
