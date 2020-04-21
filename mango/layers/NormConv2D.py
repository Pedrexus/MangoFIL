import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation


class NormalizationActivation(Layer):

    def __init__(self, activation=None, **kwargs):
        super().__init__()
        self.norm = BatchNormalization(**kwargs)
        if activation:
            try:
                self.activation = Activation(activation)
            except KeyError:
                self.activation = activation
        else:
            self.activation = tf.nn.relu

    def call(self, inputs, *args, **kwargs):
        x = self.norm(inputs)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'batchNormalization': self.norm, 'activation': self.activation})
        return config


class NormConv2D(NormalizationActivation):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(filters, kernel_size, strides, padding, use_bias=use_bias, activation=None)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = super().call(x, *args, **kwargs)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'conv2D': self.conv})
        return config
