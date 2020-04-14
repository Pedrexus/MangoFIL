from iModeling.imports import *


def create_model_mlp(x_train, y_train, x_test, y_test):

    IMGSHAPE = x_train.shape[1:]
    n_classes = len(set(y_train).union(set(y_test)))

    # -------- HYPERPARAMETERS -------- #

    dense_1 = {{choice([64, 128, 256, 384, 512, 768])}}
    dropout_1 = {{uniform(0, 1)}}
    activation_1 = {{choice(['tanh', 'sigmoid'])}}

    optimizer = Adam
    learning_rate = {{choice([1e-3, 2e-3, 5e-3])}}

    # -------- --------------- -------- #

    input_kwargs = dict(input_shape=IMGSHAPE,
                        data_format='channels_last')

    model = Sequential()

    model.add(Flatten(**input_kwargs))
    model.add(Dense(dense_1, activation=activation_1,
                    bias_initializer='zeros',
                    kernel_initializer='glorot_uniform'))
    model.add(Dropout(dropout_1))

    model.add(Dense(n_classes, activation='softmax'))

    batch_size = int(np.ceil(2**10 / dense_1))
    batch_size = batch_size if batch_size < x_train.shape[0] else None
    return compile_and_fit(model, optimizer, learning_rate, x_train, y_train,
                           batch_size)


def create_model_lenet5(x_train, y_train, x_test, y_test):
    IMGSHAPE = x_train.shape[1:]
    N_CLASSES = len(set(y_train).union(set(y_test)))

    # -------- HYPERPARAMETERS -------- #

    # 1st
    filters_1 = {{choice([6, 8, 10])}}
    kernel_size_1 = {{choice([1, 3, 5, 7])}}

    # 2nd
    filters_2 = {{choice([16, 18])}}
    kernel_size_2 = {{choice([1, 3, 5, 7])}}

    # 3rd
    dense_1 = {{choice([8, 16, 32, 64])}}
    dense_2 = {{choice([4, 8, 16, 32])}}
    dropout_1 = {{uniform(0, 1)}}

    optimizer = {{choice([Adam, Nadam])}}
    learning_rate = {{choice([1e-3, 2e-3, 5e-3])}}

    # -------- --------------- -------- #

    input_kwargs = dict(input_shape=IMGSHAPE,
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

    # 3rd layer - fully connected
    model.add(Flatten())
    model.add(Dense(dense_1, activation='tanh'))
    model.add(Dense(dense_2, activation='tanh'))
    model.add(Dropout(dropout_1))

    model.add(Dense(N_CLASSES, activation='softmax'))

    batch_size = 16
    return compile_and_fit(model, optimizer, learning_rate, x_train, y_train,
                           batch_size)


def create_model_alexnet(x_train, y_train, x_test, y_test):
    IMGSHAPE = x_train.shape[1:]
    N_CLASSES = len(set(y_train).union(set(y_test)))

    # -------- HYPERPARAMETERS -------- #
    # 1st and 2nd Convolutional Layers
    filters_1 = {{choice([4, 8, 16, 20])}}
    kernel_size_1 = {{choice([7, 9, 11, 13])}}

    # 3rd, 4th and 5th Convolutional Layers
    filters_2 = {{choice([16, 24, 32, 48])}}
    kernel_size_2 = {{choice([1, 3, 5])}}

    # 6th Fully Connected Layer
    dense_6 = {{choice([32, 64, 128, 256, 512])}}

    # 7th Fully Connected Layer
    dense_7 = {{choice([32, 64, 128, 256, 512])}}

    # 7th Fully Connected Layer
    dense_8 = {{choice([16, 32, 64, 128, 256])}}

    optimizer = Adam
    learning_rate = {{choice([2e-2, 2e-3, 2e-4])}}

    # -------- --------------- -------- #

    input_kwargs = dict(input_shape=IMGSHAPE,
                        data_format='channels_last')

    model = Sequential()

    # 1st Convolutional Layer
    model.add(
        Conv2D(filters=filters_1,
               kernel_size=kernel_size_1, strides=1,
               padding='valid', activation='relu',
               **input_kwargs))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=filters_1,
                     kernel_size=kernel_size_1, strides=1,
                     padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

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
    model.add(BatchNormalization())

    # 6th Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(dense_6, activation='relu'))
    model.add(Dropout(0.4))

    # 7th Fully Connected Layer
    model.add(Dense(dense_7, activation='relu'))
    model.add(Dropout(0.4))

    # 8th Fully Connected Layer
    model.add(Dense(dense_8, activation='relu'))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(N_CLASSES, activation='softmax'))

    batch_size = 32
    return compile_and_fit(model, optimizer, learning_rate, x_train, y_train,
                           batch_size)


def create_model_vggnet16(x_train, y_train, x_test, y_test):
    IMGSHAPE = x_train.shape[1:]
    N_CLASSES = len(set(y_train).union(set(y_test)))

    # -------- HYPERPARAMETERS -------- #
    # 1st Convolutional Layers
    filters_1 = {{choice([1, 2, 3, 4])}}

    # 2nd Convolutional Layers
    filters_2 = {{choice([2, 4, 6, 8])}}

    # 3rd Convolutional Layers
    filters_3 = {{choice([4, 6, 8, 10])}}

    # 4th Convolutional Layers
    filters_4 = {{choice([6, 8, 12, 16])}}

    # 5th Convolutional Layers
    filters_5 = {{choice([8, 16, 20, 24])}}

    # 6th Fully Connected Layer
    dense_6 = {{choice([16, 32, 64, 128, 256])}}

    # 7th Fully Connected Layer
    dense_7 = {{choice([16, 32, 64, 128, 256])}}
    dropout_7 = {{uniform(0, 1)}}

    optimizer = Adam
    learning_rate = 2e-3

    # -------- --------------- -------- #

    input_kwargs = dict(input_shape=IMGSHAPE,
                        data_format='channels_last')

    model = Sequential([

        # 1st Convolutional Layer
        Conv2D(filters=filters_1,
               kernel_size=3, strides=1,
               padding='same', activation='relu',
               **input_kwargs),
        Conv2D(filters=filters_1,
               kernel_size=3, strides=1,
               padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),

        # 2nd Convolutional Layer
        Conv2D(filters=filters_2,
               kernel_size=3, strides=1,
               padding='same', activation='relu',
               **input_kwargs),
        Conv2D(filters=filters_2,
               kernel_size=3, strides=1,
               padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),

        # 3rd Convolutional Layer
        Conv2D(filters=filters_3,
               kernel_size=3, strides=1,
               padding='same', activation='relu',
               **input_kwargs),
        Conv2D(filters=filters_3,
               kernel_size=3, strides=1,
               padding='same', activation='relu'),
        Conv2D(filters=filters_3,
               kernel_size=3, strides=1,
               padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),

        # 4th Convolutional Layer
        Conv2D(filters=filters_4,
               kernel_size=3, strides=1,
               padding='same', activation='relu',
               **input_kwargs),
        Conv2D(filters=filters_4,
               kernel_size=3, strides=1,
               padding='same', activation='relu'),
        Conv2D(filters=filters_4,
               kernel_size=3, strides=1,
               padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),

        # 5th Convolutional Layer
        Conv2D(filters=filters_5,
               kernel_size=3, strides=1,
               padding='same', activation='relu',
               **input_kwargs),
        Conv2D(filters=filters_5,
               kernel_size=3, strides=1,
               padding='same', activation='relu'),
        Conv2D(filters=filters_5,
               kernel_size=3, strides=1,
               padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),

        # 6th Fully Connected Layer
        Flatten(),
        Dense(dense_6, activation='relu'),
        BatchNormalization(),

        # 7th Fully Connected Layer
        Dense(dense_7, activation='relu'),
        Dropout(dropout_7),

        # output
        Dense(N_CLASSES, activation='softmax')
    ])

    batch_size = 16
    return compile_and_fit(model, optimizer, learning_rate, x_train, y_train,
                           batch_size)


def create_model_custom(x_train, y_train, x_test, y_test):
    IMGSHAPE = x_train.shape[1:]
    n_classes = len(set(y_train).union(set(y_test)))

    # -------- HYPERPARAMETERS -------- #

    # 1st Convolutional Layer
    filters_1 = {{choice([8, 9, 10, 12])}}
    kernel_size_1 = {{choice([7, 9, 11, 13])}}

    # 2nd Convolutional Layer
    filters_2 = {{choice([14, 16, 20, 24])}}
    kernel_size_2 = {{choice([15, 17, 19, 21])}}
    activation_2 = {{choice(['relu', 'tanh', 'sigmoid'])}}

    # 3rd Fully Connected Layer
    dense_1 = {{choice([8, 16, 32, 64, 128])}}
    dropout_1 = {{uniform(0, 1)}}
    activation_3 = {{choice(['tanh', 'sigmoid'])}}

    optimizer = Nadam
    learning_rate = 1e-4

    # -------- --------------- -------- #

    input_kwargs = dict(input_shape=IMGSHAPE,
                        data_format='channels_last')

    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=filters_1,
                     kernel_size=kernel_size_1,
                     strides=2,
                     activation='relu',
                     padding='valid',
                     **input_kwargs))
    model.add(Conv2D(filters=filters_1 // 2,
                     kernel_size=kernel_size_1 // 2,
                     strides=1,
                     activation='relu',
                     padding='valid'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=filters_2,
                     kernel_size=kernel_size_2,
                     strides=2,
                     activation=activation_2,
                     padding='same'))
    model.add(Conv2D(filters=filters_2 // 2,
                     kernel_size=kernel_size_2 // 2,
                     strides=1,
                     activation=activation_2,
                     padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    # 3rd Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(dense_1, activation=activation_3))
    model.add(Dropout(dropout_1))

    model.add(Dense(n_classes, activation='softmax'))

    batch_size = 16
    return compile_and_fit(model, optimizer, learning_rate, x_train, y_train,
                           batch_size)


def get_best_score(result):
    best_epoch = np.argmax(
        np.abs(result.history['val_f1_score']))  # loss is negative

    train_loss = result.history['loss'][best_epoch]
    t_f1 = result.history['f1_score'][best_epoch]
    t_acc = result.history['acc'][best_epoch]

    val_loss = result.history['val_loss'][best_epoch]
    v_f1 = result.history['val_f1_score'][best_epoch]
    v_acc = result.history['val_acc'][best_epoch]

    print(f'train: loss={train_loss:.3f}, f1={t_f1:.3f}, acc={t_acc:.3f} \n'
          f'val: loss={val_loss:.3f}, f1={v_f1:.3f}, acc={v_acc:.3f} \n'
          f'number of epochs: {best_epoch + 1} \n')

    # maybe loss to assert overfitting and not f1
    return v_f1 if v_f1 <= t_f1 else 0


def compile_and_fit(model, optimizer, learning_rate, x_train, y_train,
                    batch_size):
    from sklearn.model_selection import train_test_split
    from keras_metrics import sparse_categorical_f1_score
    from hyperopt import STATUS_OK
    from keras.preprocessing.image import ImageDataGenerator

    # use sparse_categorical_crossentropy on not one_hot encoded data
    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=[sparse_categorical_f1_score(), 'accuracy'],
                  optimizer=optimizer(lr=learning_rate))

    x_tr, x_valid, y_tr, y_valid = train_test_split(x_train, y_train,
                                                    stratify=y_train,
                                                    test_size=.25,
                                                    random_state=123)

    dataaug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                                 width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.15,
                                 horizontal_flip=True, fill_mode="nearest")
    train_gen = dataaug.flow(x_tr, y_tr, batch_size=batch_size)

    result = model.fit_generator(train_gen,
                                 validation_data=(x_valid, y_valid),
                                 epochs=50,
                                 steps_per_epoch=len(x_tr) // batch_size,
                                 verbose=0,
                                 shuffle=True)

    score = get_best_score(result)

    return {'loss': -score, 'status': STATUS_OK, 'model': model}
