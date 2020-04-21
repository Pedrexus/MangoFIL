from tensorflow.keras.layers import Layer, AveragePooling2D

from ...convolution.NormConv2D import NormConv2D


class TransitionBlock2D(Layer):

    def __init__(self, reduction: float, *args, **kwargs):
        """A building block for a dense block.

        :param growth_rate: float, growth rate at dense layers.
        """
        super().__init__()
        self.conv = lambda n: NormConv2D(int(n * reduction), 1, use_bias=False, *args, **kwargs)
        self.pool = AveragePooling2D(pool_size=2, strides=2)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs.shape[-1])(inputs)  # channels last
        x = self.pool(x)
        return x
