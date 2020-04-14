import numpy as np
import tensorflow as tf
from hyperas import optim
from hyperopt import tpe, Trials

from iLoad.fetch import fetch_data
# activate GPUs
from iModeling.functions import save_results
from iModeling.models import create_model_lenet5
from iNeural.printing import print_results

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def data():
    IMGSIZE = 32

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    x_train, y_train, x_test, y_test = fetch_data(IMGSIZE,
                                                  mode='A',
                                                  test_size=.3,
                                                  seed=123)

    # x_train = rgb2gray(x_train).reshape((-1, IMGSIZE, IMGSIZE, 1))
    # x_test = rgb2gray(x_test).reshape((-1, IMGSIZE, IMGSIZE, 1))

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    from hyperas.distributions import choice, uniform
    from hyperopt import STATUS_OK
    from keras import Sequential
    from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
    from keras.optimizers import Adam, SGD, Nadam
    from keras_metrics import sparse_categorical_f1_score

    best_run, best_model = optim.minimize(model=create_model_lenet5,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=200,
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
                 dataset_name='Antrac_2',
                 architecture_name='LeNet-5',
                 notes='acc=F1-score')
    print('Process completed')
