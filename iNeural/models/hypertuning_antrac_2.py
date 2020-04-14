import numpy as np
import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import STATUS_OK, tpe, Trials
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, \
    BatchNormalization
from keras.optimizers import Adam, SGD, Nadam

from iLoad.fetch import fetch_data
from iNeural.printing import print_results

# activate GPUs
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

"""Função que retorna os dados de treinamento e teste conforme
     a biblioteca Hyperas exige.

    :return: x_train, y_train, x_test, y_test
"""


def data():
    IMGSIZE = 32
    x_train, y_train, x_test, y_test = fetch_data(IMGSIZE,
                                                  mode='A',
                                                  test_size=.3,
                                                  seed=123)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    IMGSIZE = 32
    N_CLASSES = 2

    # -------- HYPERPARAMETERS -------- #

    dense_1 = {{choice([64, 128, 256, 512, 1024])}}
    dropout_1 = {{uniform(0, 1)}}
    activation_1 = {{choice(['relu', 'tanh', 'sigmoid'])}}

    optimizer = {{choice([Adam, SGD, Nadam])}}
    learning_rate = {{choice([1e-2, 1e-3, 1e-4, 1e-5])}}

    # -------- --------------- -------- #

    input_kwargs = dict(input_shape=(IMGSIZE, IMGSIZE, 3),
                        data_format='channels_last')

    model = Sequential()

    model.add(Flatten(**input_kwargs))
    model.add(Dense(dense_1, activation=activation_1))
    model.add(Dropout(dropout_1))

    model.add(Dense(N_CLASSES, activation='softmax'))

    # use sparse_categorical_crossentropy on not one_hot encoded data
    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=optimizer(lr=learning_rate))

    result = model.fit(x_train, y_train,
                       epochs=20,
                       verbose=0,
                       validation_split=.25,
                       shuffle=True)

    best_epoch = np.argmin(np.abs(result.history['val_loss']))  # loss is negative

    train_loss = result.history['loss'][best_epoch]
    t_acc = result.history['acc'][best_epoch]

    val_loss = result.history['val_loss'][best_epoch]
    v_acc = result.history['val_acc'][best_epoch]

    print(f'train: loss={train_loss:.3f}, acc={t_acc:.3f} \n'
          f'val: loss={val_loss:.3f}, acc={v_acc:.3f} \n'
          f'number of epochs: {best_epoch + 1} \n')

    return {'loss': val_loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
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

"""Resultados preliminares (pre-bootstrapping)

valid_split = .45: 
    conv_layer = True
    filters = 8
    kernel_size = 7 
    strides = 1
    activation_1 = 'sigmoid'
    padding = 'same'
    pool_size = 2
    dropout_0 = .07369

    dense_1 = 64
    dense_2 = 128
    activation_2 = 'sigmoid'
    dropout_1 = .45986
    dropout_2 = .53822
    batch_size = 4
    extra_layer = True
"""
