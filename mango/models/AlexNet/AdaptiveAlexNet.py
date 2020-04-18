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
        _n, _s = self.define_params()

        self.conv1 = Conv2D(_n(96, 2), kernel_size=_s(11, 3), strides=_s(4, 2), activation=tf.nn.relu, **self._)
        self.pool1 = MaxPooling2D(pool_size=_s(3, 2), strides=_s(2, 2), padding='valid')
        self.conv2 = Conv2D(_n(256, 2), kernel_size=_s(5, 3), strides=_s(1, 1), activation=tf.nn.relu, padding='same')
        self.pool2 = MaxPooling2D(pool_size=_s(3, 2), strides=_s(2, 2), padding='valid')
        self.conv3 = Conv2D(_n(384, 2), kernel_size=_s(3, 3), strides=_s(1, 1), activation=tf.nn.relu, padding='same')
        self.conv4 = Conv2D(_n(384, 2), kernel_size=_s(3, 3), strides=_s(1, 1), activation=tf.nn.relu, padding='same')
        self.conv5 = Conv2D(_n(256, 2), kernel_size=_s(3, 3), strides=_s(1, 1), activation=tf.nn.relu, padding='same')
        self.pool5 = MaxPooling2D(pool_size=_s(3, 2), strides=_s(2, 2), padding='valid')
        self.dense1 = Dense(_n(4096, 2), activation=tf.nn.relu)
        self.dense2 = Dense(_n(4096, 2), activation=tf.nn.relu)