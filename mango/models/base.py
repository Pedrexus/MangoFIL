from functools import lru_cache
from math import log2, log10

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense


class BaseModel(Model):
    N_CLASSES = None
    INPUT_SHAPE = ()

    def __init__(self, n_classes=N_CLASSES, input_shape=INPUT_SHAPE, *args, **kwargs):
        """base keras model for easy using of summary

        :param n_classes: number of distinct classes
        :param input_shape: tuple (x, y, z) of image shape
        """
        super().__init__(*args, **kwargs)

        self._n_classes, self._input_shape = n_classes, input_shape
        self._ = dict(input_shape=input_shape, data_format='channels_last')

        self.out = Dense(n_classes, activation=tf.nn.softmax)

    def define_params(self):
        """define params of adaptive version of model

        the params are defined with a LINEAR scale based
        that each layer has an output shape = (None, x, y, z)
        where x, y depends on input_shape
        and z depends on n_classes

        cls.N_CLASSES: original n_classes implementation
        cls.INPUT_SHAPE: original input_shape implementation

        :param n_classes: actual n_classes
        :param input_shape: actual input_shape
        :return: function, function
        """
        N, n = self.N_CLASSES, self._n_classes
        S, s = min(self.INPUT_SHAPE[:-1]), min(self._input_shape[:-1])

        @lru_cache(maxsize=1024)
        def kernel(a: int, b: int) -> int:
            _a = (a * s) // S
            return max(_a, b)

        @lru_cache(maxsize=1024)
        def units(a: int, b: int) -> int:
            _a = (a * n) // N
            return max(_a, b) * kernel(int(log10(a)), 1)  # or kernel(log2(a), 1)

        return units, kernel

    def model(self):
        x = Input(shape=self._input_shape)
        return Model(inputs=[x], outputs=self.call(x))

    def summary(self, *args, **kwargs):
        return self.model().summary(*args, **kwargs)
