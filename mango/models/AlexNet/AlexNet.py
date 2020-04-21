import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

from ...layers.convolution import DoubleConv2D
from ...layers.dense import DoubleDense
from ..base import BaseModel
from ..registry import Registry


class AlexNet(BaseModel, Registry):

    N_CLASSES = 1000
    INPUT_SHAPE = (227, 227, 3)

    def __init__(self, n_classes=N_CLASSES, input_shape=INPUT_SHAPE, *args, **kwargs):
        """Original AlexNet implementation

        from the original paper:
            n_classes = 1000
            input_shape = (227, 227, 3)
        """
        super().__init__(n_classes, input_shape, *args, **kwargs)

        self.conv1 = Conv2D(96, kernel_size=11, strides=4, activation=tf.nn.relu, **self._)
        self.pool1 = MaxPooling2D(pool_size=3, strides=2, padding='valid')
        self.conv2 = Conv2D(256, kernel_size=5, strides=1, activation=tf.nn.relu, padding='same')
        self.pool2 = MaxPooling2D(pool_size=3, strides=2, padding='valid')
        self.conv3a = DoubleConv2D(384, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.conv3b = Conv2D(256, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.pool3 = MaxPooling2D(pool_size=3, strides=2, padding='valid')
        self.flatten = Flatten()
        self.dense1 = DoubleDense(4096, activation=tf.nn.relu)

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.out(x)
        return x

if __name__ == "__main__":
    model = AlexNet(1000, (227, 227, 3))
