import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from .VGGNet16 import VGGNet16
from ..registry import Registry


class AdaptiveVGGNet16(VGGNet16, Registry):

    def __init__(self, n_classes, input_shape, *args, **kwargs):
        """Adaptive VGGNet16 implementation"""
        super().__init__(n_classes, input_shape, *args, **kwargs)
        _n, _k = self.define_params()

        self.conv1a = Conv2D(_n(64, 2), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same', **self._)
        self.conv1b = Conv2D(_n(64, 2), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')

        self.conv2a = Conv2D(_n(128, 4), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv2b = Conv2D(_n(128, 4), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')

        self.conv3a = Conv2D(_n(256, 8), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv3b = Conv2D(_n(256, 8), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv3c = Conv2D(_n(256, 8), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')

        self.conv4a = Conv2D(_n(512, 16), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv4b = Conv2D(_n(512, 16), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv4c = Conv2D(_n(512, 16), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')

        self.conv5a = Conv2D(_n(512, 16), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv5b = Conv2D(_n(512, 16), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv5c = Conv2D(_n(512, 16), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')

        self.dense1 = Dense(_n(4096, 64), activation=tf.nn.relu)
        self.dense2 = Dense(_n(4096, 64), activation=tf.nn.relu)
