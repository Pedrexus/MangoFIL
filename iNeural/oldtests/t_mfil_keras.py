import tensorflow as tf

from iLoad.fetch import fetch_data
from iNeural.printing import print_results

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# -------- HYPERPARAMETERS -------- #
IMGSIZE = 64  # px
LEARNING_RATE = 1e-4
N_CLASSES = 2
N_EPOCHS = 20
# -------- --------------- -------- #

x_train, y_train, x_test, y_test = fetch_data(IMGSIZE, mode='A')

training_set = (x_train, y_train)
test_set = (x_test, y_test)

model_1 = tf.keras.models.Sequential([
    # 1st layer: Conv_1
    tf.keras.layers.Conv2D(input_shape=(IMGSIZE, IMGSIZE, 3),
                           filters=8,
                           kernel_size=5, strides=1,
                           padding='VALID', activation='relu',
                           bias_initializer='zeros',
                           data_format='channels_last'),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='SAME'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(N_CLASSES, activation='softmax')
])


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(IMGSIZE, IMGSIZE, 3)),
    tf.keras.layers.Dense(2048, activation='sigmoid'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(N_CLASSES, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_set[0], training_set[1],
          validation_split=.1, shuffle=True,
          epochs=N_EPOCHS, verbose=2)

print(f'Results after {N_EPOCHS} epochs:')
print_results(model, training_data=training_set, test_data=test_set)

# classes = {n: Counter(eval(f'{n}_set[1]')) for n in
#            ['training', 'validation', 'test']}
