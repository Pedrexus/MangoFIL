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
        _n, _s = self.define_params()

        # self.conv1a.filters = _n(self.conv1a.filters, 2)
        # self.conv1a.kernel_size = _s(self.conv1a.kernel_size, 2)

        self.conv1a = Conv2D(_n(128, 2), kernel_size=_s(3, 2), strides=1, activation=tf.nn.relu, padding='valid', **self._)
        self.conv1b = Conv2D(_n(128, 2), kernel_size=_s(3, 2), strides=1, activation=tf.nn.relu, padding='valid')
        self.pool1 = MaxPooling2D(pool_size=_s(2, 2), strides=2, padding='valid')

        self.conv2a = Conv2D(_n(256, 4), kernel_size=_s(3, 2), strides=1, activation=tf.nn.relu, padding='valid')
        self.conv2b = Conv2D(_n(256, 4), kernel_size=_s(3, 2), strides=1, activation=tf.nn.relu, padding='valid')
        self.pool2 = MaxPooling2D(pool_size=_s(2, 2), strides=2, padding='valid')

        self.conv3a = Conv2D(_n(384, 6), kernel_size=_s(3, 2), strides=1, activation=tf.nn.relu, padding='valid')
        self.conv3b = Conv2D(_n(256, 4), kernel_size=_s(3, 2), strides=1, activation=tf.nn.relu, padding='valid')
        self.pool3 = MaxPooling2D(pool_size=_s(2, 2), strides=2, padding='valid')

        self.dense1 = Dense(_n(512, 8), activation=tf.nn.relu)
