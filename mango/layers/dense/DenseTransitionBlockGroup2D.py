from typing import Iterable

from tensorflow.keras.layers import Layer

from mango.layers.dense.blocks.DenseTransitionBlock2D import DenseTransitionBlock2D


class DenseTransitionBlockGroup2D(Layer):

    def __init__(self, growth_rate: int, blocks: Iterable[int], reduction: float, *args, **kwargs):
        super().__init__()
        self.blocks = [DenseTransitionBlock2D(growth_rate, n, reduction, *args, **kwargs) for n in blocks]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for block in self.blocks:
            x = block(x, *args, **kwargs)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'blocks': self.blocks})
        return config
