import tensorflow as tf

from .VGGNet16 import VGGNet16
from ..registry import Registry
from ...layers.convolution import DoubleConv2D, TripleConv2D
from ...layers.dense import DoubleDense


class AdaptiveVGGNet16(VGGNet16, Registry):

    def __init__(self, n_classes, input_shape, *args, **kwargs):
        """Adaptive VGGNet16 implementation"""
        super().__init__(n_classes, input_shape, *args, **kwargs)
        _n, _k = self.define_params()

        self.conv1 = DoubleConv2D(_n(64, 16), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv2 = DoubleConv2D(_n(128, 32), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv3 = TripleConv2D(_n(256, 64), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv4 = TripleConv2D(_n(512, 128), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.conv5 = TripleConv2D(_n(512, 128), kernel_size=_k(3, 3), strides=1, activation=tf.nn.relu, padding='same')
        self.dense1 = DoubleDense(_n(4096, 512), activation=tf.nn.relu)
