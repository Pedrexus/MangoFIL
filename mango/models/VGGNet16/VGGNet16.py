import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Flatten

from ..base import BaseModel
from ..registry import Registry
from ...layers.convolution.DoubleConv2D import DoubleConv2D
from ...layers.convolution.TripleConv2D import TripleConv2D
from ...layers.dense.DoubleDense import DoubleDense


class VGGNet16(BaseModel, Registry):
    N_CLASSES = 1000
    INPUT_SHAPE = (224, 224, 3)

    def __init__(self, n_classes=N_CLASSES, input_shape=INPUT_SHAPE, *args, **kwargs):
        """Original VGGNet16 implementation

        from the original paper:
            n_classes = 1000
            input_shape = (224, 224, 3)
        """
        super().__init__(n_classes, input_shape, *args, **kwargs)

        self.conv1 = DoubleConv2D(64, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.pool1 = MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv2 = DoubleConv2D(128, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.pool2 = MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv3 = TripleConv2D(256, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.pool3 = MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv4 = TripleConv2D(512, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.pool4 = MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv5 = TripleConv2D(512, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.pool5 = MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.flatten = Flatten()
        self.dense1 = DoubleDense(4096, activation=tf.nn.relu)

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.out(x)
        return x


if __name__ == "__main__":
    model = VGGNet16(1000, (224, 224, 3))
