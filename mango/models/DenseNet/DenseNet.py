import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dropout

from ..base import BaseModel
from ..registry import Registry
from ...layers.convolution.NormConv2D import NormalizationActivation
from ...layers.convolution.PadNormConv2D import PadNormConv2D
from ...layers.dense import DenseTransitionBlockGroup2D
from ...layers.dense.blocks import DenseBlock2D


class DenseNet(BaseModel, Registry):
    N_CLASSES = 1000
    INPUT_SHAPE = (224, 224, 3)

    def __init__(self, n_classes=N_CLASSES, input_shape=INPUT_SHAPE, blocks=(6, 12, 24, 16), reduction=.5, growth_rate=32,
                 pooling=None, n_dropout=0, epsilon=1.001e-5, *args,
                 **kwargs):
        """Densely Connected Convolutional Networks
            (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)
            # Reference implementation
                - [Torch DenseNets] : https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua
                - [TensorNets]  : https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py

        :param n_classes:
        :param input_shape:
        :param blocks: numbers of building blocks for the four dense layers.
        :param reduction: float, compression rate at transition layers.
        :param include_top: whether to include the fully-connected layer at the top of the network.
        :param pooling: optional pooling mode for feature extraction
            - `None` - no pooling is applied before dense softmax.
            - `avg` - global average pooling is applied before dense softmax.
            - `max` - global max pooling will is applied before dense softmax.
        """
        super().__init__(n_classes, input_shape, *args, **kwargs)
        self.n_dropout = n_dropout
        assert self.n_dropout < 3 and isinstance(self.n_dropout, int)

        self.conv1 = PadNormConv2D(64, kernel_size=7, strides=2, activation=tf.nn.relu, padding='valid', epsilon=epsilon)
        self.pool1 = MaxPooling2D(pool_size=3, strides=2, padding='valid')
        self.dropout1 = Dropout(.5)
        self.dtgroup = DenseTransitionBlockGroup2D(growth_rate, blocks[:-1], reduction, epsilon=epsilon)
        self.dropout2 = Dropout(.5)
        self.dblock2 = DenseBlock2D(growth_rate, blocks[-1], epsilon=epsilon)
        self.norm2 = NormalizationActivation(activation=tf.nn.relu, epsilon=epsilon)

        if pooling == 'avg':
            self.pool = GlobalAveragePooling2D()
        elif pooling == 'max':
            self.pool = GlobalMaxPooling2D()
        else:
            self.pool = None

        self.flatten = Flatten()

    def call(self, inputs, training=False, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        if training and self.n_dropout >= 1:
            x = self.dropout1(x, training)
        x = self.dtgroup(x)
        if training and self.n_dropout >= 2:
            x = self.dropout2(x, training)
        x = self.dblock2(x)
        x = self.norm2(x)
        if self.pool:
            x = self.pool(x)
        x = self.flatten(x)
        x = self.out(x)
        return x
