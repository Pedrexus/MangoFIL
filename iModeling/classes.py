from collections import defaultdict
from datetime import datetime
from functools import partial

import numpy as np
from keras.engine.saving import load_model
from keras.models import clone_model
from keras_metrics import sparse_categorical_f1_score
from pandas import DataFrame
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score, \
    average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from uncertainties import ufloat

from iLoad.fetch import load_norm_center_data


class Evaluator:
    custom_objects = {'sparse_categorical_f1_score':
                          sparse_categorical_f1_score()}

    def __init__(self, model_path, mode):
        self.model = load_model(model_path, custom_objects=self.custom_objects)
        self.__model_copy = self.deepcopy_model(self.model)

        self.imgsize = self.model.input_shape[1]
        self.n_classes = self.model.output_shape[1]
        self.mode = mode

        self.x_data, self.y_data = load_norm_center_data(self.imgsize, mode)

        self.bootstrap_built = False
        self.cvs_built = set()
        self.model_reset = False

        self.results = {}

    def reset_results(self):
        self.bootstrap_built = False
        self.cvs_built = set()

        self.results = {}

    @staticmethod
    def deepcopy_model(model):
        model_copy = clone_model(model)
        model_copy.build(model.input_shape)
        model_copy.compile(optimizer=model.optimizer, loss=model.loss)
        model_copy.set_weights(model.get_weights())

        return model_copy

    def reset_model(self):
        new_model = self.reset_weights(self.model)
        self.model = new_model

        self.model_reset = True

    @classmethod
    def reset_weights(cls, model):
        model_copy = cls.deepcopy_model(model)

        from keras import backend as K
        session = K.get_session()
        for layer in model_copy.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

        return model_copy

    def bootstrap(self, test_size, bootstrap_epochs=20, train_epochs=20,
                  batch_size=32):

        train, test = defaultdict(list), defaultdict(list)
        progress_bar = tqdm(range(bootstrap_epochs), smoothing=.1)

        for i in progress_bar:
            if len(train) and len(test):
                progress_bar.set_postfix(
                    **{'avg loss': np.mean(test[log_loss.__name__]),
                       'avg f1': np.mean(test[f1_score.__name__])})

            # -------- BOOTSTRAPPING -------- #
            seed = datetime.now().microsecond

            x_train, x_test, \
            y_train, y_test = train_test_split(self.x_data, self.y_data,
                                               stratify=self.y_data,
                                               test_size=test_size,
                                               random_state=seed)

            # ------------ METRICS ----------- #
            metrics_values = self.evaluate_model(self.model,
                                                 x_train, y_train,
                                                 x_test, y_test,
                                                 n_epochs=train_epochs,
                                                 batch_size=batch_size)

            train[log_loss.__name__].append(metrics_values[0][0])
            test[log_loss.__name__].append(metrics_values[0][1])

            for (train_m, test_m), m in zip(metrics_values[1:], self.metrics):
                train[m.__name__].append(train_m)
                test[m.__name__].append(test_m)

            # ------------ -------- ----------- #

        self.results['bootstrap'] = dict(
            bootstrap_epochs=bootstrap_epochs,
            train_epochs=train_epochs,
            train=train,
            test=test,
        )

        self.bootstrap_built = True

    def cv(self, n_splits, train_epochs=20, batch_size=32):
        skf = StratifiedKFold(n_splits=n_splits,
                              random_state=datetime.now().microsecond,
                              shuffle=True)

        train, test = defaultdict(list), defaultdict(list)
        progress_bar = tqdm(skf.split(self.x_data, self.y_data),
                            smoothing=.1, total=n_splits)

        for train_index, test_index in progress_bar:
            if len(train) and len(test):
                progress_bar.set_postfix(
                    **{'avg loss': np.mean(test[log_loss.__name__]),
                       'avg f1': np.mean(test[f1_score.__name__])})

            # -------- CROSS VALIDATION -------- #

            x_train, x_test = self.x_data[train_index], self.x_data[test_index]
            y_train, y_test = self.y_data[train_index], self.y_data[test_index]

            # ------------ METRICS ----------- #
            metrics_values = self.evaluate_model(self.model,
                                                 x_train, y_train,
                                                 x_test, y_test,
                                                 n_epochs=train_epochs,
                                                 batch_size=batch_size)

            train[log_loss.__name__].append(metrics_values[0][0])
            test[log_loss.__name__].append(metrics_values[0][1])

            for (train_m, test_m), m in zip(metrics_values[1:], self.metrics):
                train[m.__name__].append(train_m)
                test[m.__name__].append(test_m)

            # ------------ -------- ----------- #

        self.results[f'cv_{n_splits}'] = dict(
            n_splits=n_splits,
            train_epochs=train_epochs,
            train=train,
            test=test,
        )

        self.cvs_built.add(n_splits)

    @property
    def log_loss(self):
        metric = log_loss.__name__
        return self.get_metric_df(metric)

    f1 = partial(f1_score, average='weighted')
    f1.__name__ = 'f1_score'

    metrics = [
        accuracy_score,
        f1,
        # roc_auc_score,
        # average_precision_score,  # AP summarizes pre_rec_curve
        confusion_matrix,
    ]

    @property
    def acc(self):
        metric = accuracy_score.__name__
        return self.get_metric_df(metric)

    @property
    def f1(self):
        metric = f1_score.__name__
        return self.get_metric_df(metric)

    @property
    def auc_roc(self):
        metric = roc_auc_score.__name__
        return self.get_metric_df(metric)

    @property
    def ap(self):
        metric = average_precision_score.__name__
        return self.get_metric_df(metric)

    @property
    def confusion_matrix(self):
        metric = confusion_matrix.__name__

        def get_metric(data_name, metric_name):
            train = self.results[data_name]['train'][metric_name]
            test = self.results[data_name]['test'][metric_name]

            m_train = np.mean(train, axis=0)
            m_test = np.mean(test, axis=0)

            return {'train': m_train, 'test': m_test}

        ev_names = [f'cv_{n}' for n in self.cvs_built]
        if self.bootstrap_built:
            ev_names.append('bootstrap')

        values = {ev: get_metric(ev, metric) for ev in ev_names}

        return values

    def get_metric_df(self, metric_name):

        ev_names = [f'cv_{n}' for n in self.cvs_built]
        if self.bootstrap_built:
            ev_names.append('bootstrap')

        values = DataFrame({ev: self.get_metric(ev, metric_name)
                            for ev in ev_names})
        values.index.name = metric_name

        return values

    def get_metric(self, data_name, metric_name):
        train = self.results[data_name]['train'][metric_name]
        test = self.results[data_name]['test'][metric_name]

        m_train, m_test = self.get_ufloat(train), self.get_ufloat(test)

        return {'train': m_train, 'test': m_test}

    @staticmethod
    def get_ufloat(metric_list):
        avg_metric = np.mean(metric_list)
        std_metric = float('{:0.1e}'.format(np.std(metric_list)))
        return ufloat(avg_metric, std_metric)

    @classmethod
    def model_predict(cls, model, x_train, y_train, x_test, n_epochs,
                      batch_size=32):
        new_model = cls.reset_weights(model)

        new_model.fit(x_train, y_train, batch_size=batch_size, shuffle=True,
                      epochs=n_epochs, verbose=0)

        y_train_prob = new_model.predict(x_train, verbose=0)
        y_train_pred = y_train_prob.argmax(axis=-1)

        y_test_prob = new_model.predict(x_test, verbose=0)
        y_test_pred = y_test_prob.argmax(axis=-1)

        return y_train_prob, y_train_pred, y_test_prob, y_test_pred

    @classmethod
    def evaluate_metrics(cls, y_train_true, y_train_pred,
                         y_test_true, y_test_pred):

        metrics_values = []
        for m in cls.metrics:
            train_m = m(y_train_true, y_train_pred)
            test_m = m(y_test_true, y_test_pred)
            metrics_values.append((train_m, test_m))

        return metrics_values

    @classmethod
    def evaluate_model(cls, model, x_train, y_train, x_test, y_test, n_epochs,
                       batch_size=32):
        y_train_prob, \
        y_train_pred, \
        y_test_prob, \
        y_test_pred = cls.model_predict(model,
                                        x_train, y_train, x_test,
                                        n_epochs, batch_size)

        train_loss = log_loss(y_train, y_train_prob)
        test_loss = log_loss(y_test, y_test_prob)

        metrics_values = [(train_loss, test_loss)]

        metrics_values += cls.evaluate_metrics(y_train, y_train_pred,
                                               y_test, y_test_pred)

        return metrics_values


if __name__ == '__main__':
    model_path = r'D:\Coding\Python\MangoFIL\iModeling\saved_models\Antrac_Collapse_4_VGGNet-16.h5'
    e = Evaluator(model_path, mode='AC')

    e.reset_results()
    e.cv(10, train_epochs=10)
    # e.bootstrap(.3, bootstrap_epochs=20, train_epochs=40)  # , e.cv(2), e.cv(4), e.cv(10)

    print(e.log_loss)
    print(e.f1)
    print(e.acc)
    # print(e.auc_roc)
    # print(e.ap)
    # print(e.confusion_matrix)
