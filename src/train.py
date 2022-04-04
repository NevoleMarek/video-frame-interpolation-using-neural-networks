###
#
#   Deprecated, Not complete
#   Used training code is in train.ipynb.
#   Used this script to train on my computer.
#   Quickly moved to google colab because of training times.
#
###

import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from config import WEIGHTS_PATH
from generator import DataGenerator
from models import small_test, u_net, u_net_bn

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
LOAD = '.\weights\w-001-0.28.hdf5'

def main():
    loss = 'mse'
    optimizer = Adam(learning_rate=LEARNING_RATE)
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(WEIGHTS_PATH,"unetbn-{loss:.2f}.hdf5"),
            monitor = 'loss',
            save_best_only = True,
            mode = 'min',
            save_freq = 'epoch'
        ),
        EarlyStopping(
            monitor = 'loss',
            patience = 3
        )
    ]

    model = u_net_bn()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    #if LOAD:
    #    model.load_weights(LOAD)
    model.fit(
        x = DataGenerator(BATCH_SIZE),
        epochs = EPOCHS,
        callbacks = callbacks,
        use_multiprocessing = True,
        workers = 2,
        verbose = 1
    )

if __name__ == '__main__':
    main()
