from tensorflow.keras.utils import Sequence
import numpy as np
class DataGenerator(Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data['click_seq']) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = {}
        for key in self.data.keys():
            batch_data[key] = self.data[key][idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_data