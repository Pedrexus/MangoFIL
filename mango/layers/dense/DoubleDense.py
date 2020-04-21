from tensorflow.keras.layers import Layer, Dense


class DoubleDense(Layer):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dense1 = Dense(*args, **kwargs)
        self.dense2 = Dense(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'dense1': self.dense1, 'dense2': self.dense2})
        return config
