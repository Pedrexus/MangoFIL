import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense

from .LeNet5 import LeNet5
from ..registry import Registry


class AdaptiveLeNet5(LeNet5, Registry):

    def __init__(self, n_classes, input_shape, *args, **kwargs):
        """Adaptive LeNet5 implementation"""
        super().__init__(n_classes, input_shape, *args, **kwargs)
        _n, _k = self.define_params()

        self.conv1 = Conv2D(_n(6, 6), kernel_size=_k(5, 2), strides=1, activation=tf.nn.tanh, padding='valid', **self._)
        self.pool1 = AveragePooling2D(pool_size=2, strides=2, padding='same')
        self.conv2 = Conv2D(_n(6, 6), kernel_size=_k(5, 2), strides=1, activation=tf.nn.tanh, padding='valid')
        self.pool2 = AveragePooling2D(pool_size=2, strides=2, padding='same')
        self.dense1 = Dense(_n(120, 20), activation=tf.nn.tanh)
        self.dense2 = Dense(_n(84, 14), activation=tf.nn.tanh)
