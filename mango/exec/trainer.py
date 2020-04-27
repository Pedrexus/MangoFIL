from copy import deepcopy
from functools import lru_cache

import tensorflow as tf
from numpy import array, unique, mean, std, argmax, absolute
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from ..helpers import one_hot_encode


class Trainer:

    def __init__(self, x: array, y: array, model: tf.keras.Model, test_size: float, validation_size: float = 0,
                 random_state: int = 123, db=None, **kwargs):
        self.x = x
        self.y = y

        self.model = model
        self.model_kwargs = kwargs
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

        self.db = db

    @lru_cache(maxsize=2)
    def preprocessing(self, one_hot=False):
        x = self.x.astype('float32')

        # ~ min-max normalization
        x /= 255

        mean_vals = mean(x, axis=0)
        std_val = std(x)

        # gauss normalization
        x = (x - mean_vals) / std_val

        if one_hot:
            # one-hot encoding
            y = one_hot_encode(self.y.reshape(-1, 1))
        else:
            y = self.y

        return x, y

    @lru_cache(maxsize=2)
    def splitting(self, *args, **kwargs):
        x, y = self.preprocessing(*args, **kwargs)

        x, x_test, y, y_test = train_test_split(
            x, y, stratify=y, test_size=self.test_size, random_state=self.random_state
        )
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, stratify=y, test_size=self.validation_size, random_state=self.random_state
        )

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def __model(self) -> tf.keras.Model:
        return self.model(n_classes=unique(self.y).shape[0], input_shape=self.x.shape[1:], **self.model_kwargs)

    @staticmethod
    def __data_augmentation(augmentation, x_train, y_train, x_valid=None, y_valid=None):
        augmentation = deepcopy(augmentation)
        batch_size = augmentation.pop("batch_size", 32)
        n_images = augmentation.pop("n_images", len(x_train) * 2)

        # number of images = batch_size * steps_per_epoch (per epoch)
        steps_per_epoch = max(n_images // batch_size, len(x_train) // batch_size, 1)

        aug = ImageDataGenerator(**augmentation)
        train_gen = aug.flow(x_train, y_train, batch_size=batch_size)

        if x_valid is not None and y_valid is not None:
            validation_steps = max(len(x_valid) // batch_size, 1)
            valid_gen = aug.flow(x_valid, y_valid, batch_size=batch_size)

            return train_gen, dict(steps_per_epoch=steps_per_epoch, validation_data=valid_gen, validation_steps=validation_steps)
        return train_gen, dict(steps_per_epoch=steps_per_epoch)

    def train(self, loss, metrics, optimizer: tf.keras.optimizers.Optimizer, augmentation, *args, **kwargs):
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.splitting(one_hot=True)

        model = self.__model()
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        if augmentation:
            train_gen, aug_kwargs = self.__data_augmentation(augmentation, x_train, y_train, x_valid, y_valid)
            result = model.fit(train_gen, *args, **aug_kwargs, **kwargs)
        else:
            result = model.fit(x_train, y_train, *args, validation_data=(x_valid, y_valid), **kwargs)

        if self.db:
            document = self.make_train_document(loss, optimizer, result, augmentation)
            self.db.insert('mango-train', document)
        return result

    def make_train_document(self, loss, optimizer, result, augmentation):
        best_epoch = argmax(absolute(result.history['val_f1_score']))  # loss is negative
        t_f1 = result.history['f1_score'][best_epoch]
        v_f1 = result.history['val_f1_score'][best_epoch]

        document = dict(
            n_classes=len({*self.y}),
            input_shape=self.x.shape[1:],
            data_size=self.x.shape[0],
            model=str(self.model.__name__),
            model_kwargs=self.model_kwargs,
            loss=str(loss),
            optimizer=str(optimizer),
            test_size=float(self.test_size),
            validation_size=float(self.validation_size),
            random_state=int(self.random_state),
            augmentation={**augmentation} if augmentation else None,
            best_epoch=int(best_epoch),
            best_train_score=float(t_f1),
            best_validation_score=float(v_f1),
            result=result.history,
        )

        return document

    def cv(self, n_splits, loss, metrics, optimizer, augmentation, *args, **kwargs):
        x, y = self.preprocessing(one_hot=False)

        all_results = []

        # --------- K-FOLD CROSS VALIDATION ---- #
        skf = StratifiedKFold(n_splits, True, self.random_state)
        for train_index, test_index in tqdm(skf.split(x, y), total=n_splits, smoothing=.1):

            # ----------- VARIABLES ------------ #
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # ------------ ENCODING ------------ #
            # tensorflow F1Score demands one-hot encoding
            y_train = one_hot_encode(y_train.reshape(-1, 1))
            y_test = one_hot_encode(y_test.reshape(-1, 1))

            # ------------- MODEL -------------- #
            model = self.__model()
            model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

            # ----------- TRAINING ------------- #
            if augmentation:
                train_gen, train_aug_kwargs = self.__data_augmentation(augmentation, x_train, y_train)
                result = model.fit(train_gen, *args, **train_aug_kwargs, **kwargs)
            else:
                result = model.fit(x_train, y_train, *args, **kwargs)

            # ----------- EVALUATION ----------- #
            verbose = kwargs.get('verbose', 0)
            verbose = 1 if verbose == 2 else verbose
            if augmentation:
                test_gen, test_aug_kwargs = self.__data_augmentation(augmentation, x_test, y_test)
                evaluation = model.evaluate(test_gen, verbose=verbose, steps=test_aug_kwargs['steps_per_epoch'])
            else:
                evaluation = model.evaluate(x_test, y_test, verbose=verbose)

            all_results.append([result.history, evaluation])

        if self.db:
            document = self.make_cv_document(n_splits, loss, optimizer, augmentation, all_results)
            self.db.insert('mango-cv', document)
        return all_results

    def make_cv_document(self, n_splits, loss, optimizer, augmentation, all_results):
        n_classes, input_shape = len({*self.y}), self.x.shape[1:]

        document = dict(
            n_classes=n_classes,
            input_shape=input_shape,
            data_size=self.x.shape[0],
            model=str(self.model.__name__),
            model_kwargs=self.model_kwargs,
            loss=str(loss),
            optimizer=str(optimizer),
            test_size=float(self.test_size),
            validation_size=float(self.validation_size),
            random_state=int(self.random_state),
            augmentation={**augmentation} if augmentation else None,
            n_splits=n_splits,
            result=all_results,
        )

        return document

    def gpu_info(self):
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            print('GPU device not found')
        else:
            print('Found GPU at: {}'.format(device_name))
