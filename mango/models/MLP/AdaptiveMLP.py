import tensorflow as tf
from tensorflow.keras.layers import Dense

from .MLP import MLP
from ..registry import Registry


class AdaptiveMLP(MLP, Registry):

    def __init__(self, n_classes, input_shape, *args, **kwargs):
        """Adaptive LeNet5 implementation"""
        super().__init__(n_classes, input_shape, *args, **kwargs)
        _n, _s = self.define_params()

        self.dense1 = Dense(_n(1024, 4), activation=tf.nn.sigmoid)
