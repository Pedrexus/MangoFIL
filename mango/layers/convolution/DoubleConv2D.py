from tensorflow.keras.layers import Layer, Conv2D


class DoubleConv2D(Layer):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1 = Conv2D(*args, **kwargs)
        self.conv2 = Conv2D(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'conv1': self.conv1, 'conv2': self.conv2})
        return config
