import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import STATUS_OK, tpe, Trials
from keras import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, \
    BatchNormalization
from keras.optimizers import Adam

from iLoad.fetch import fetch_data

from iNeural.printing import print_results

# activate GPUs
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def data():

    IMGSIZE = 64
    x_train, y_train, x_test, y_test = fetch_data(IMGSIZE, mode='A',
                                                  test_size=.003,
                                                  seed=123)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):

    IMGSIZE = 64
    N_CLASSES = 2
    MIN_TRAIN_ACC = .9

    # -------- HYPERPARAMETERS -------- #

    conv_layer = {{choice([False, True])}}
    filters = {{choice([4, 8, 16, 32])}}
    kernel_size = {{choice([3, 5, 7])}}
    strides = {{choice([1, 2])}}
    activation_1 = {{choice(['relu', 'tanh', 'sigmoid'])}}
    padding = {{choice(['valid', 'same'])}}
    pool_size = {{choice([2, 4])}}
    dropout_0 = {{uniform(0, 1)}}

    dense_1 = {{choice([64, 128, 256, 512])}}
    dropout_1 = {{uniform(0, 1)}}
    activation_2 = {{choice(['relu', 'tanh', 'sigmoid'])}}

    extra_layer = {{choice([False, True])}}
    dense_2 = {{choice([64, 128, 256, 512, 1024])}}
    dropout_2 = {{uniform(0, 1)}}

    # -------- --------------- -------- #

    input_kwargs = dict(input_shape=(IMGSIZE, IMGSIZE, 3),
                        data_format='channels_last')

    model = Sequential()

    if conv_layer:
        model.add(Conv2D(filters,
                         kernel_size=kernel_size, strides=strides,
                         activation=activation_1, padding=padding,
                         **input_kwargs))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout_0))

        model.add(Flatten())
    else:
        model.add(Flatten(**input_kwargs))

    model.add(Dense(dense_1, activation=activation_2))
    model.add(Dropout(dropout_1))

    if extra_layer:
        model.add(Dense(dense_2, activation=activation_2))
        model.add(Dropout(dropout_2))

    model.add(Dense(N_CLASSES, activation='softmax'))

    # use sparse_categorical_crossentropy on not one_hot encoded data
    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=Adam(lr=0.001))

    result = model.fit(x_train, y_train,
                       epochs=20,
                       verbose=0,
                       validation_split=.45,
                       shuffle=True)

    validation_acc = result.history['val_acc'][-1]
    training_acc = result.history['acc'][-1]
    acc = validation_acc if training_acc > MIN_TRAIN_ACC else .01
    print('Last validation acc of epoch:', validation_acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


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
