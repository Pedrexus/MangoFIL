import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation


class NormConv2D(Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), activation=None, padding='valid', **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(filters, kernel_size, strides, padding, activation=None)
        self.norm = BatchNormalization()
        if activation:
            try:
                self.activation = Activation(activation)
            except KeyError:
                self.activation = activation
        else:
            self.activation = tf.nn.relu

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'conv2D': self.conv, 'batchNormalization': self.norm, 'activation': self.activation})
        return config
