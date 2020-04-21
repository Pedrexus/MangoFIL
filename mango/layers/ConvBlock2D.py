from tensorflow.keras.layers import Layer, Conv2D, Concatenate, AveragePooling2D

from mango.layers.NormConv2D import NormalizationActivation, NormConv2D


class ConvBlock2D(Layer):

    def __init__(self, growth_rate, *args, **kwargs):
        """A building block for a dense block.

        :param growth_rate: float, growth rate at dense layers.
        """
        super().__init__()
        self.norm = NormalizationActivation(*args, **kwargs)
        self.normconv1 = NormConv2D(4 * growth_rate, 1, use_bias=False)
        self.conv2 = Conv2D(growth_rate, 3, padding='same', use_bias=False)
        self.concat = Concatenate()

    def call(self, inputs, *args, **kwargs):
        x = self.norm(inputs)
        x = self.normconv1(x)
        x = self.conv2(x)
        x = self.concat([inputs, x])
        return x


