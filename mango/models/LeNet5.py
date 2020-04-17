import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Dropout
from .registry import Registry

class LeNet5(tf.keras.Model, Registry):

  def __init__(self, n_classes, input_shape):
    super().__init__()
    self.conv1 = Conv2D(6, kernel_size=5, strides=1, activation=tf.nn.tanh, padding='valid', input_shape=input_shape, data_format='channels_last')
    self.pool1 = AveragePooling2D(pool_size=2, strides=2, padding='same')
    self.conv2 = Conv2D(6, kernel_size=5, strides=1, activation=tf.nn.tanh, padding='valid')
    self.pool2 = AveragePooling2D(pool_size=2, strides=2, padding='same')
    self.dense1 = Dense(120, activation=tf.nn.tanh)
    self.dense1 = Dense(84, activation=tf.nn.tanh)
    self.dense2 = Dense(n_classes, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.dense1(x)
    return self.dense2(x)

