from functools import lru_cache
from math import log2, log10, sqrt

import tensorflow as tf
from numpy import prod
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

        the params are defined with a LINEAR/SQRT scale based
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
        S, s = prod(self.INPUT_SHAPE), prod(self._input_shape)

        @lru_cache(maxsize=1024)
        def kernel(k: int, b: int) -> int:
            """defines adapted kernel size

            S^2 / k^2 = cte

            :param k: original kernel size
            :param b: min kernel size
            :return: adapted kernel size
            """
            a = k * k * self.INPUT_SHAPE[-1]
            _a = ((a * s) // S) ** (1 / 3)
            return max(int(_a), b)

        @lru_cache(maxsize=1024)
        def units(u: int, b: int) -> int:
            """defines adapted unit amount
            (dense units or filter units)

            :param u: original unit amount
            :param b: min unit amount
            :return: adapted units amount
            """
            a = u * kernel(int(log10(u)), 1)
            _a = (a * n) // N
            return max(_a, b)

        return units, kernel

    def model(self):
        x = Input(shape=self._input_shape)
        return Model(inputs=[x], outputs=self.call(x))

    def summary(self, *args, **kwargs):
        return self.model().summary(*args, **kwargs)
