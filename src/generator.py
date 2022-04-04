import os
import random

import numpy as np
import tensorflow.keras as keras

from config import PARTITION_SIZE, PARTITIONS_PATH


class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.partitions = os.listdir(PARTITIONS_PATH)
        self.n_batches = self.compute_len()

    def __len__(self):
        """ Return number of batches. """
        return self.n_batches

    def __getitem__(self, idx):
        """ Return batch with index 'idx'. """
        ratio = int(PARTITION_SIZE/self.batch_size)
        part_id = idx // ratio
        batch_id = idx % ratio
        batch_start = batch_id * self.batch_size
        batch_end = batch_start + self.batch_size
        part = np.load(os.path.join(PARTITIONS_PATH,self.partitions[part_id]))
        if (idx+1) == self.n_batches:
            return (np.array(part['X'][batch_start:])-127.5) / 127.5, \
                    (np.array(part['Y'][batch_start:])-127.5) / 127.5
        else:
            return (np.array(part['X'][batch_start:batch_end])-127.5) / 127.5, \
                    (np.array(part['Y'][batch_start:batch_end])-127.5) / 127.5

    def on_epoch_end(self):
        random.shuffle(self.partitions)


    def compute_len(self):
        """ Computer expected number of batches from video files. """
        ln = (len(self.partitions) - 1) * int(PARTITION_SIZE/self.batch_size)
        last_part = np.load(os.path.join(PARTITIONS_PATH,self.partitions[-1]))
        sz = np.shape(last_part['X'])[0]
        return int(ln + np.ceil(sz/self.batch_size))

def main():
    """ TEST """
    g = DataGenerator(512)
    x, y = g[len(g)-1]
    print(np.shape(x))

if __name__ == '__main__':
    main()
