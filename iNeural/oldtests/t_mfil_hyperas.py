import keras
import tensorflow as tf
from hyperas import optim
from hyperas.distributions import uniform
from hyperopt import STATUS_OK, tpe, Trials
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, \
    AveragePooling2D
from keras.layers.core import Dense, Dropout
from keras.models import Sequential

from iLoad.fetch import fetch_data
from iNeural.printing import print_results

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# -------- HYPERPARAMETERS -------- #
IMGSIZE = 256  # px
LEARNING_RATE = 1e-4
N_CLASSES = 2
N_EPOCHS = 20


# -------- --------------- -------- #


def data():
    x_train, y_train, x_test, y_test = fetch_data(64, mode='A')
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    imgsize = 64
    N_CLASSES = 2

    N_FILTERS = 4

    KERNEL_SIZE_1 = 5
    KERNEL_SIZE_2 = 3
    CONV_STRIDES = 1
    PADDING_1 = 'valid'
    PADDING_2 = 'valid'

    POOL_SIZE_1 = 3
    POOL_SIZE_2 = 2

    ACTIVATION_1 = 'relu'
    ACTIVATION_2 = 'sigmoid'

    model = Sequential()

    # 1st layer: Conv1
    model.add(Conv2D(N_FILTERS, input_shape=(imgsize, imgsize, 3),
                     kernel_size=KERNEL_SIZE_1, strides=CONV_STRIDES,
                     activation=ACTIVATION_1, padding=PADDING_1,
                     data_format='channels_last'))
    model.add(AveragePooling2D(pool_size=POOL_SIZE_1))

    # 2nd layer: Conv2
    model.add(Conv2D(N_FILTERS * 2,
                     kernel_size=KERNEL_SIZE_2, strides=CONV_STRIDES,
                     activation=ACTIVATION_1, padding=PADDING_2))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=POOL_SIZE_2))

    # 3rd layer: Conv3
    model.add(Conv2D(N_FILTERS * 4,
                     kernel_size=KERNEL_SIZE_2, strides=CONV_STRIDES,
                     activation=ACTIVATION_1, padding=PADDING_2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=POOL_SIZE_2))

    # 4th layer: Conv4
    model.add(Conv2D(N_FILTERS * 8,
                     kernel_size=KERNEL_SIZE_2, strides=CONV_STRIDES,
                     activation=ACTIVATION_1, padding=PADDING_2))
    model.add(BatchNormalization())

    # 4th layer: FC1
    model.add(Flatten())
    # 512
    model.add(Dense(1024, activation=ACTIVATION_2))
    model.add(BatchNormalization())
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(N_CLASSES, activation='softmax'))
    adam = keras.optimizers.Adam(lr=0.001)

    # use sparse_categorical_crossentropy one not one_hot encoded
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=adam)

    result = model.fit(x_train, y_train,
                       batch_size=32,
                       epochs=20,
                       verbose=2,
                       validation_split=.1)

    validation_acc = result.history['val_acc'][-1]
    training_acc = result.history['acc'][-1]
    acc = validation_acc if training_acc > .75 else .01
    print('Last validation acc of epoch:', validation_acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = data()

    print("Results of best performing model:")
    print_results(best_model,
                  training_data=(X_train, Y_train),
                  test_data=(X_test, Y_test))

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
