import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense

from .AlexNet import AlexNet
from ..registry import Registry


class AdaptiveAlexNet(AlexNet, Registry):

    def __init__(self, n_classes, input_shape, *args, **kwargs):
        """Adaptive AlexNet implementation

        each layer has an output of shape (None, x, y, z)
        where x, y depends on input_shape
        and z depends on n_classes
        """
        super().__init__(n_classes, input_shape, *args, **kwargs)
        _n, _k = self.define_params()

        self.conv1 = Conv2D(_n(96, 6), kernel_size=_k(11, 3), strides=4, activation=tf.nn.relu, **self._)

        self.conv2 = Conv2D(_n(256, 16), kernel_size=_k(5, 3), strides=1, activation=tf.nn.relu, padding='same')

        self.conv3a = Conv2D(_n(384, 24), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv3b = Conv2D(_n(384, 24), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv3c = Conv2D(_n(256, 16), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')

        self.dense1 = Dense(_n(4096, 64), activation=tf.nn.relu)
        self.dense2 = Dense(_n(4096, 64), activation=tf.nn.relu)