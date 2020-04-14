import numpy as np
import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import STATUS_OK, tpe, Trials
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD, Nadam

from iLoad.fetch import fetch_data
from iNeural.printing import print_results

# activate GPUs
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def data():
    IMGSIZE = 256
    x_train, y_train, x_test, y_test = fetch_data(IMGSIZE,
                                                  mode='C',
                                                  test_size=.3,
                                                  seed=123)
    print("\n\nMode C: Only Collapse")
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    IMGSIZE = x_train.shape[1]
    N_CLASSES = 2

    # -------- HYPERPARAMETERS -------- #
    filters_1 = {{choice([4, 8, 16])}}
    kernel_size_1 = {{choice([7, 9, 11])}}

    filters_2 = {{choice([16, 32, 64])}}
    kernel_size_2 = {{choice([1, 3])}}

    optimizer = Nadam
    learning_rate = 1e-3

    # -------- --------------- -------- #

    input_kwargs = dict(input_shape=(IMGSIZE, IMGSIZE, 3),
                        data_format='channels_last')

    model = Sequential()

    # 1st Convolutional Layer
    model.add(
        Conv2D(filters=filters_1,
               kernel_size=kernel_size_1, strides=2,
               padding='valid', activation='relu',
               **input_kwargs))
    model.add(MaxPooling2D(pool_size=2))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=filters_1,
                     kernel_size=kernel_size_1, strides=2,
                     padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=filters_2,
                     kernel_size=kernel_size_2, strides=1,
                     padding='valid', activation='relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=filters_2,
                     kernel_size=kernel_size_2, strides=1,
                     padding='valid', activation='relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=filters_2,
                     kernel_size=kernel_size_2, strides=1,
                     padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    # 6th Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))

    # 7th Fully Connected Layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))

    # 8th Fully Connected Layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(N_CLASSES, activation='softmax'))

    # use sparse_categorical_crossentropy on not one_hot encoded data
    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=optimizer(lr=learning_rate))

    result = model.fit(x_train, y_train,
                       batch_size=64,
                       epochs=40,
                       verbose=0,
                       validation_split=.25,
                       shuffle=True)

    best_epoch = np.argmin(
        np.abs(result.history['val_loss']))  # loss is negative

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
                                          max_evals=50,
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = data()

    print("Results of best performing model:")
    print_results(best_model,
                  training_data=(X_train, Y_train),
                  test_data=(X_test, Y_test))

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
