import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input

from ..base import BaseModel
from ..registry import Registry


class SINGHNet(BaseModel, Registry):

    N_CLASSES = 4
    INPUT_SHAPE = (128, 128, 3)

    def __init__(self, n_classes=N_CLASSES, input_shape=INPUT_SHAPE, *args, **kwargs):
        """Original SINGHNet implementation from paper
        Multilayer Convolution Neural Network for the Classification of Mango Leaves Infected by Anthracnose Disease
        by UDAY PRATAP SINGH et al.

        This implementation uses padding='valid' and pool_strides=2 so that the number of params is reduced
        """
        super().__init__(n_classes, input_shape, *args, **kwargs)

        self.conv1a = Conv2D(128, kernel_size=3, strides=1, activation=tf.nn.relu, padding='valid', **self._)
        self.conv1b = Conv2D(128, kernel_size=3, strides=1, activation=tf.nn.relu, padding='valid')
        self.pool1 = MaxPooling2D(pool_size=2, strides=2, padding='valid')
        self.drop1 = Dropout(.5)

        self.conv2a = Conv2D(256, kernel_size=3, strides=1, activation=tf.nn.relu, padding='valid')
        self.conv2b = Conv2D(256, kernel_size=3, strides=1, activation=tf.nn.relu, padding='valid')
        self.pool2 = MaxPooling2D(pool_size=2, strides=2, padding='valid')
        self.drop2 = Dropout(.5)

        self.conv3a = Conv2D(384, kernel_size=3, strides=1, activation=tf.nn.relu, padding='valid')
        self.conv3b = Conv2D(256, kernel_size=3, strides=1, activation=tf.nn.relu, padding='valid')
        self.pool3 = MaxPooling2D(pool_size=2, strides=2, padding='valid')
        self.drop3 = Dropout(.2)

        self.flatten = Flatten()
        self.dense1 = Dense(512, activation=tf.nn.relu)

    def call(self, inputs, training=None, *args, **kwargs):
        x = self.conv1a(inputs)
        x = self.conv1b(x)
        x = self.pool1(x)
        if training:
            x = self.drop1(x, training)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool2(x)
        if training:
            x = self.drop2(x, training)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)
        if training:
            x = self.drop3(x, training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.out(x)
        return x
