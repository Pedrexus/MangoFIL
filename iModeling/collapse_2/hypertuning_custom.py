import numpy as np
import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import STATUS_OK, tpe, Trials
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, \
    BatchNormalization
from keras.optimizers import Adam, SGD, Nadam
from keras.utils import np_utils
from keras_metrics import binary_f1_score, sparse_categorical_f1_score
from sklearn.model_selection import train_test_split
from iLoad.fetch import fetch_data
from iModeling.functions import save_results
from iModeling.models import create_model_mlp, get_best_score, \
    create_model_custom
from iNeural.metrics import f1_score
from iNeural.printing import print_results

# activate GPUs
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def data():
    IMGSIZE = 512
    x_train, y_train, x_test, y_test = fetch_data(IMGSIZE,
                                                  mode='AC',
                                                  test_size=.3,
                                                  seed=123)
    print("\n\nMode C: Only Collapse")
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model_custom,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = data()

    print("Results of best performing model:")
    print_results(best_model,
                  training_data=(X_train, Y_train),
                  test_data=(X_test, Y_test))

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    print("Saving if model is better:")
    save_results(best_model, X_test, Y_test,
                 dataset_name='Collapse_2',
                 architecture_name='Custom',
                 notes=f'F1-Score, img=64px')
    print('Process completed')

