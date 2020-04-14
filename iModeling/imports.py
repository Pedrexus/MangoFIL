import numpy as np
import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import STATUS_OK
from hyperopt import tpe, Trials
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D, \
    BatchNormalization, Activation
from keras.optimizers import Adam, SGD, Nadam

from iLoad.fetch import fetch_data
from iModeling.functions import save_results
from iModeling.models import get_best_score, compile_and_fit
from iNeural.printing import print_results

# activate GPUs
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

if __name__ == '__main__':
    print(np.e, choice, uniform, STATUS_OK, Sequential, Dense, Dropout,
          Flatten, MaxPooling2D, Conv2D, BatchNormalization, Adam, SGD, Nadam,
          optim, tpe, Trials, fetch_data, get_best_score, compile_and_fit,
          print_results, save_results)
