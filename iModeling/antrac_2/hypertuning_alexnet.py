import numpy as np
import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import STATUS_OK, tpe, Trials
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD, Nadam

from iLoad.fetch import fetch_data
from iModeling.functions import save_results
from iModeling.models import create_model_alexnet, get_best_score
from iNeural.printing import print_results
from keras_metrics import sparse_categorical_f1_score

# activate GPUs
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def data():
    IMGSIZE = 256
    x_train, y_train, x_test, y_test = fetch_data(IMGSIZE,
                                                  mode='A',
                                                  test_size=.3,
                                                  seed=123)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model_alexnet,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = data()

    print("Results of best performing model:")
    print_results(best_model,
                  training_data=(X_train, Y_train),
                  test_data=(X_test, Y_test))

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    save_results(best_model, X_test, Y_test,
                 dataset_name='Antrac_2',
                 architecture_name='AlexNet',
                 notes='acc=F1-score')
    print('Process completed')