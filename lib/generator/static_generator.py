import pandas as pd
import os
import sys


class StaticExchange:
    """A static instrument exchange, in which the price history is loaded from a csv file"""

    def __init__(self, **kwargs):
        self.data_frame = pd.DataFrame()
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.file_name = os.path.join(parent_dir, 'data', 'dataset', 'ADANIPORTS-EQ.csv')
        self.reset()

    def load_csv(self):
        if os.path.exists(self.file_name):
            self.data_frame = pd.read_csv(self.file_name)
            self.data_frame['date'] = pd.to_datetime(self.data_frame['date'], format='%d-%m-%Y %I:%M:%S %p')
            self.data_frame['date'] = self.data_frame['date'].astype(str)
            self.data_frame = self.data_frame.sort_values(['date'])
            self.data_frame = self.data_frame.set_index('date')
        else:
            print("File not Found!")

    def reset(self):
        self.data_frame = pd.DataFrame()
        self.load_csv()

