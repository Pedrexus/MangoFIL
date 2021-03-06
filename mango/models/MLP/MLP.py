import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense

from ..base import BaseModel
from ..registry import Registry


class MLP(BaseModel, Registry):

    N_CLASSES = 4
    INPUT_SHAPE = (28, 28, 1)

    def __init__(self, n_classes=N_CLASSES, input_shape=INPUT_SHAPE, *args, **kwargs):
        """MLP implementation"""
        super().__init__(n_classes, input_shape, *args, **kwargs)

        self.flatten = Flatten(**self._)
        self.dense1 = Dense(1024, activation=tf.nn.sigmoid)

    def call(self, inputs, *args, **kwargs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.out(x)
        return x
