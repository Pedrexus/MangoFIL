from tensorflow.keras.layers import Layer

from mango.layers.dense.blocks.DenseBlock2D import DenseBlock2D
from mango.layers.dense.blocks.TransitionBlock2D import TransitionBlock2D


class DenseTransitionBlock2D(Layer):

    def __init__(self, growth_rate: int, blocks: int, reduction: float, *args, **kwargs):
        super().__init__()
        self.dblock = DenseBlock2D(growth_rate, blocks, *args, **kwargs)
        self.tblock = TransitionBlock2D(reduction, *args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        x = self.dblock(inputs)
        x = self.tblock(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'dense_block': self.dblock, 'transition_block': self.tblock})
        return config
