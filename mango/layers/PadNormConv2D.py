from tensorflow.keras.layers import ZeroPadding2D

from .NormConv2D import NormConv2D


class PadNormConv2D(NormConv2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad1 = ZeroPadding2D(padding=((3, 3), (3, 3)))
        self.conv.use_bias = False
        self.pad2 = ZeroPadding2D(padding=((1, 1), (1, 1)))

    def call(self, inputs, *args, **kwargs):
        x = self.pad1(inputs)
        x = super().call(x, *args, **kwargs)
        x = self.pad2(x)
        return x
