from functools import lru_cache

from numpy import array, unique, mean, std, argmax, absolute, log10, log2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from ..helpers import one_hot_encode


class Trainer:

    def __init__(self, x: array, y: array, model: tf.keras.Model, test_size: float, validation_size: float = 0,
                 random_state: int = 123, db=None):
        self.x = x
        self.y = y

        self.model = model
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

        self.db = db

    @lru_cache(maxsize=2)
    def preprocessing(self):
        x = self.x.astype('float32')

        # ~ min-max normalization
        x /= 255

        mean_vals = mean(x, axis=0)
        std_val = std(x)

        # gauss normalization
        x = (x - mean_vals) / std_val

        # one-hot encoding
        y = one_hot_encode(self.y.reshape(-1, 1))

        x, x_test, y, y_test = train_test_split(
            x, y, stratify=y, test_size=self.test_size, random_state=self.random_state
        )
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, stratify=y, test_size=self.validation_size, random_state=self.random_state
        )

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def train(self, augmentation, loss, metrics, optimizer: tf.keras.optimizers.Optimizer, *args, **kwargs):
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.preprocessing()

        model = self.model(n_classes=unique(self.y).shape[0], input_shape=self.x.shape[1:])
        model.compile(
            loss=loss,
            metrics=metrics,
            optimizer=optimizer
        )

        if augmentation:
            batch_size = augmentation.pop("batch_size", 32)

            steps_per_epoch = min(len(x_train) // batch_size or 1, 2)
            validation_steps = min(len(x_valid) // batch_size or 1, 2)

            aug = ImageDataGenerator(**augmentation)
            train_gen = aug.flow(x_train, y_train, batch_size=batch_size)
            valid_gen = aug.flow(x_valid, y_valid, batch_size=batch_size)

            result = model.fit(
                train_gen,
                *args,
                steps_per_epoch=steps_per_epoch,
                validation_data=valid_gen,
                validation_steps=validation_steps,
                **kwargs,
            )
        else:
            batch_size = None
            result = model.fit(x_train, y_train, *args, validation_data=(x_valid, y_valid), **kwargs)

        if self.db:
            document = self.make_document(augmentation, batch_size, loss, optimizer, result)
            self.db.insert(document)
        return result

    def make_document(self, augmentation, batch_size, loss, optimizer, result):
        best_epoch = argmax(absolute(result.history['val_f1_score']))  # loss is negative
        t_f1 = result.history['f1_score'][best_epoch]
        v_f1 = result.history['val_f1_score'][best_epoch]

        document = dict(
            n_classes=len({*self.y}),
            input_shape=self.x.shape[1:],
            data_size=self.x.shape[0],
            model=str(self.model),
            loss=str(loss),
            optimizer=str(optimizer),
            test_size=float(self.test_size),
            validation_size=float(self.validation_size),
            random_state=int(self.random_state),
            augmentation={'batch_size': batch_size, **augmentation},
            best_epoch=int(best_epoch),
            best_train_score=float(t_f1),
            best_validation_score=float(v_f1),
            result=result.history,
        )

        return document

    def gpu_info(self):
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            print('GPU device not found')
        else:
            print('Found GPU at: {}'.format(device_name))
