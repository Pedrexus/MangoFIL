from tensorflow.keras.layers import Layer, Concatenate

from mango.layers.NormConv2D import NormActConv2D


class ConvBlock2D(Layer):

    def __init__(self, growth_rate, *args, **kwargs):
        """A building block for a dense block.

        :param growth_rate: float, growth rate at dense layers.
        """
        super().__init__()
        self.normconv1 = NormActConv2D(4 * growth_rate, kernel_size=1, use_bias=False *args, **kwargs)
        self.normconv2 = NormActConv2D(growth_rate, kernel_size=3, use_bias=False, padding='same', *args, **kwargs)
        self.concat = Concatenate()

    def call(self, inputs, *args, **kwargs):
        x = self.normconv1(inputs)
        x = self.normconv2(x)
        x = self.concat([inputs, x])
        return x
