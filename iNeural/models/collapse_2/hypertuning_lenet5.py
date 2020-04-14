import numpy as np
import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import STATUS_OK, tpe, Trials
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D, \
    MaxPooling2D
from keras.optimizers import Adam, SGD, Nadam
from keras_metrics import binary_precision, binary_recall, \
    sparse_categorical_f1_score, binary_f1_score
from sklearn.metrics import f1_score

from iLoad.fetch import fetch_data
# activate GPUs
from iNeural.printing import print_results

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def data():
    IMGSIZE = 128

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    x_train, y_train, x_test, y_test = fetch_data(IMGSIZE,
                                                  mode='C',
                                                  test_size=.3,
                                                  seed=123)
    print("\n\nMode C: Only Collapse")
    # x_train = rgb2gray(x_train).reshape((-1, IMGSIZE, IMGSIZE, 1))
    # x_test = rgb2gray(x_test).reshape((-1, IMGSIZE, IMGSIZE, 1))

    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    IMGSIZE = x_train.shape[1]
    N_CLASSES = 2

    # -------- HYPERPARAMETERS -------- #
    # 1st
    filters_1 = {{choice([6, 8, 10, 12, 14])}}
    kernel_size_1 = {{choice([1, 3, 5, 7])}}

    # 2nd
    filters_2 = {{choice([16, 18, 20, 24])}}
    kernel_size_2 = {{choice([1, 3, 5])}}
    dropout_1 = {{uniform(0, 1)}}

    # 3rd
    dense_1 = {{choice([8, 16, 32, 64, 128])}}
    dense_2 = {{choice([4, 8, 16, 32, 64])}}
    dropout_2 = {{uniform(0, 1)}}

    optimizer = Nadam
    learning_rate = {{choice([1e-4, 5e-4, 1e-3])}}

    # -------- --------------- -------- #

    input_kwargs = dict(input_shape=(IMGSIZE, IMGSIZE, 3),
                        data_format='channels_last')

    model = Sequential()

    # 1s layer - convolution
    model.add(Conv2D(filters=filters_1,
                     kernel_size=kernel_size_1, strides=1,
                     activation='tanh', padding='valid',
                     **input_kwargs))
    model.add(MaxPooling2D(pool_size=2))

    # 2nd layer - convolution
    model.add(Conv2D(filters=filters_2,
                     kernel_size=kernel_size_2, strides=1,
                     activation='tanh', padding='valid',
                     **input_kwargs))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_1))

    # 3rd layer - fully connected
    model.add(Flatten(**input_kwargs))
    model.add(Dense(dense_1, activation='tanh'))
    model.add(Dense(dense_2, activation='tanh'))
    model.add(Dropout(dropout_2))

    model.add(Dense(N_CLASSES, activation='softmax'))

    # use sparse_categorical_crossentropy on not one_hot encoded data
    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=[binary_f1_score()],
                  optimizer=optimizer(lr=learning_rate))

    result = model.fit(x_train, y_train,
                       batch_size=64,
                       epochs=20,
                       verbose=0,
                       validation_split=.25,
                       shuffle=True)

    best_epoch = np.argmin(
        np.abs(result.history['val_f1_score']))  # loss is negative

    train_loss = result.history['loss'][best_epoch]
    t_acc = result.history['f1_score'][best_epoch]

    val_loss = result.history['val_loss'][best_epoch]
    v_acc = result.history['val_f1_score'][best_epoch]

    print(f'train: loss={train_loss:.3f}, acc={t_acc:.3f} \n'
          f'val: loss={val_loss:.3f}, acc={v_acc:.3f} \n'
          f'number of epochs: {best_epoch + 1} \n')

    return {'loss': -v_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    trials = Trials()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=200,
                                          trials=trials)

    X_train, Y_train, X_test, Y_test = data()

    print("Results of best performing model:")
    print_results(best_model,
                  training_data=(X_train, Y_train),
                  test_data=(X_test, Y_test))

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
