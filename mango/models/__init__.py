import tensorflow as tf

from .registry import Registry

from .MLP import MLP, AdaptiveMLP
from .LeNet5 import LeNet5, AdaptiveLeNet5
from .AlexNet import AlexNet, AdaptiveAlexNet
from .VGGNet16 import VGGNet16, AdaptiveVGGNet16
from .SINGHNet import SINGHNet, AdaptiveSINGHNet


def get_model(name: str) -> tf.keras.Model:
    return Registry.get(name)
