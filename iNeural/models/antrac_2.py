import tensorflow as tf
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam

# activate GPUs
from iNeural.functions import bootstrap_model
from iNeural.printing import print_avg_results_dict

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# -------- HYPERPARAMETERS -------- #
IMGSIZE = 64  # px
N_CLASSES = 2
# -------- --------------- -------- #

input_kwargs = dict(input_shape=(IMGSIZE, IMGSIZE, 3),
                    data_format='channels_last')

model = Sequential([
    # 1st layer: FC_1
    Flatten(**input_kwargs),
    Dense(256, activation='tanh'),
    Dropout(.39),

    # output
    Dense(N_CLASSES, activation='softmax'),
])

model.compile(optimizer=Adam(lr=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', ])

if __name__ == '__main__':
    n_bootstrapping = 20

    model_results = bootstrap_model(model,
                                    imgsize=64,
                                    test_size=.5,
                                    mode='A',
                                    bootstrap_epochs=n_bootstrapping,
                                    train_epochs=20)

    print(f'\nResults after {n_bootstrapping} bootstrapping epochs:')
    print_avg_results_dict(model_results)
