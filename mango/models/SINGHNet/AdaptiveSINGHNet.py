import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input

from ..SINGHNet import SINGHNet
from ..registry import Registry


class AdaptiveSINGHNet(SINGHNet, Registry):

    def __init__(self, n_classes, input_shape, *args, **kwargs):
        """Original SINGHNet implementation from paper
        Multilayer Convolution Neural Network for the Classification of Mango Leaves Infected by Anthracnose Disease
        by UDAY PRATAP SINGH et al.
        """
        super().__init__(n_classes, input_shape, *args, **kwargs)
        _n, _k = self.define_params()

        # self.conv1a.filters = _n(self.conv1a.filters, 2)
        # self.conv1a.kernel_size = _k(self.conv1a.kernel_size, 2)

        self.conv1a = Conv2D(_n(128, 4), kernel_size=_k(3, 2), strides=1, activation=tf.nn.relu, padding='valid', **self._)
        self.conv1b = Conv2D(_n(128, 4), kernel_size=_k(3, 2), strides=1, activation=tf.nn.relu, padding='valid')

        self.conv2a = Conv2D(_n(256, 8), kernel_size=_k(3, 2), strides=1, activation=tf.nn.relu, padding='valid')
        self.conv2b = Conv2D(_n(256, 8), kernel_size=_k(3, 2), strides=1, activation=tf.nn.relu, padding='valid')

        self.conv3a = Conv2D(_n(384, 12), kernel_size=_k(3, 2), strides=1, activation=tf.nn.relu, padding='valid')
        self.conv3b = Conv2D(_n(256, 8), kernel_size=_k(3, 2), strides=1, activation=tf.nn.relu, padding='valid')

        self.dense1 = Dense(_n(512, 16), activation=tf.nn.relu)
