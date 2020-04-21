from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Layer, AveragePooling2D

from ...convolution.NormConv2D import NormConv2D


class TransitionBlock2D(Layer):

    def __init__(self, reduction: float, *args, **kwargs):
        """Transition block between dense blocks

        :param reduction: float, the compression factor that reduces the number of filters.
        """
        super().__init__()
        self.conv = lambda n: NormConv2D(int(n * reduction), 1, use_bias=False, *args, **kwargs)
        self.pool = AveragePooling2D(pool_size=2, strides=2)

    def build(self, input_shape):
        self.conv = self.conv(int(input_shape[-1]))

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)  # channels last
        x = self.pool(x)
        return x
