import pandas as pd
from sklearn.model_selection import train_test_split
from .config.config import TRAIN_TEST_SPLIT, SEED

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        return pd.read_csv(self.data_path)

class DataPreprocessor:
    def preprocess(self, data):
        # Perform preprocessing steps here
        return data

    def split_data(self, data):
        return train_test_split(data, test_size=TRAIN_TEST_SPLIT, random_state=SEED)
