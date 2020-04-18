import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, Flatten

from mango.metrics import SparseCategoricalF1Score, SparseCategoricalTruePositives
from mango.models.registry import Registry


class LeNet5(tf.keras.Model, Registry):

    def __init__(self, n_classes, input_shape):
        super().__init__()
        self.conv1 = Conv2D(6, kernel_size=5, strides=1, activation=tf.nn.tanh, padding='valid',
                            input_shape=input_shape, data_format='channels_last')
        self.pool1 = AveragePooling2D(pool_size=2, strides=2, padding='same')
        self.conv2 = Conv2D(6, kernel_size=5, strides=1, activation=tf.nn.tanh, padding='valid')
        self.pool2 = AveragePooling2D(pool_size=2, strides=2, padding='same')
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation=tf.nn.tanh)
        self.dense2 = Dense(84, activation=tf.nn.tanh)
        self.out = Dense(n_classes, activation=tf.nn.softmax)

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    import numpy as np
    import tensorflow_addons as tfa

    def one_hot_encode(arr):
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(arr)
        return enc.transform(arr.reshape(-1, 1)).toarray()

    N = 50
    n = 2
    x = np.array([np.random.rand(28, 28, 3) for i in range(N)]).astype('float32')
    y = one_hot_encode(np.random.randint(n, size=N).reshape(-1, 1)).astype('float32')

    model = LeNet5(n, x.shape[1:])

    model.compile(
        optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', tfa.metrics.F1Score(n, average='weighted')]
    )

    model.fit(x, y, shuffle=True, verbose=2, epochs=5)

    y_pred = model.predict(x)

    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = cce(y, y_pred)

