from .LeNet5 import LeNet5
from .registry import Registry
import tensorflow as tf

def get_model(name: str) -> tf.keras.Model:
    return Registry.get(name)