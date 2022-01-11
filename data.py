import pandas as pd
import numpy as np

class Data:
    def __init__(self, csv):
        self.df = pd.read_csv("data_sets/" + csv)
        self.data_list = self.df.values.tolist()

    def generate_train_test_sets(self, split=0.3):
        filter = np.random.rand(len(self.df)) <= split
        self.test_set = self.df[filter].values.tolist()
        self.training_set = self.df[~filter].values.tolist()