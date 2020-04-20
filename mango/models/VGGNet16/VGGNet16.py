import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from ..base import BaseModel
from ..registry import Registry


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

        self.conv1a = Conv2D(64, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same', **self._)
        self.conv1b = Conv2D(64, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.pool1 = MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv2a = Conv2D(128, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.conv2b = Conv2D(128, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.pool2 = MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv3a = Conv2D(256, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.conv3b = Conv2D(256, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.conv3c = Conv2D(256, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.pool3 = MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv4a = Conv2D(512, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.conv4b = Conv2D(512, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.conv4c = Conv2D(512, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.pool4 = MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.conv5a = Conv2D(512, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.conv5b = Conv2D(512, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.conv5c = Conv2D(512, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
        self.pool5 = MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.flatten = Flatten()
        self.dense1 = Dense(4096, activation=tf.nn.relu)
        self.dense2 = Dense(4096, activation=tf.nn.relu)

    def call(self, inputs, *args, **kwargs):
        x = self.conv1a(inputs)
        x = self.conv1b(x)
        x = self.pool1(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.conv3c(x)
        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.conv4c(x)
        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv5c(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


if __name__ == "__main__":
    model = VGGNet16(1000, (224, 224, 3))
