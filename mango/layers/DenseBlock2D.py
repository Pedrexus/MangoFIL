from mango.layers.ConvBlock2D import ConvBlock2D


class DenseBlock2D(ConvBlock2D):

    def __init__(self, blocks: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = blocks

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for _ in range(self.blocks):
            x = super().call(x, *args, **kwargs)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'blocks': self.blocks})
        return config
