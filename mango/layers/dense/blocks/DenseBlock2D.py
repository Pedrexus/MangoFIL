from tensorflow.keras.layers import Layer
from .ConvBlock2D import ConvBlock2D


class DenseBlock2D(Layer):

    def __init__(self, growth_rate: int, blocks: int, *args, **kwargs):
        super().__init__()
        self.blocks = [ConvBlock2D(growth_rate, *args, **kwargs) for _ in range(blocks)]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for block in self.blocks:
            x = block(x, *args, **kwargs)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'blocks': self.blocks})
        return config
