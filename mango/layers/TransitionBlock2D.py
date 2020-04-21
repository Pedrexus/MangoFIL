from tensorflow.keras.layers import Layer, Conv2D, Concatenate, AveragePooling2D

from mango.layers.NormConv2D import NormalizationActivation, NormConv2D

class TransitionBlock2D(Layer):

    def __init__(self, n_channels: int, reduction: float, *args, **kwargs):
        """A building block for a dense block.

        :param growth_rate: float, growth rate at dense layers.
        """
        super().__init__()
        self.norm = NormalizationActivation(*args, **kwargs)
        self.conv = Conv2D(int(n_channels * reduction), 1, use_bias=False)
        self.pool = AveragePooling2D(pool_size=2, strides=2)

    def call(self, inputs, *args, **kwargs):
        x = self.norm(inputs)
        x = self.conv(x)
        x = self.pool(x)
        return x
