import datetime as dt
from collections import defaultdict
from copy import deepcopy

import numpy as np
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from iLoad.fetch import load_norm_center_data


def reset_weights(model):
    from keras import backend as K
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def evaluate_model(model, x_train, y_train, x_test, y_test, n_epochs=20):
    new_model = deepcopy(model)

    reset_weights(new_model)
    new_model.fit(x_train, y_train,
                  shuffle=True,
                  epochs=n_epochs,
                  verbose=0)

    y_train_prob = new_model.predict(x_train, verbose=0)
    y_train_pred = y_train_prob.argmax(axis=-1)

    y_test_prob = new_model.predict(x_test, verbose=0)
    y_test_pred = y_test_prob.argmax(axis=-1)

    return y_train_prob, y_train_pred, y_test_prob, y_test_pred


def get_metrics(y_train_true, y_train_pred, y_test_true, y_test_pred,
                train_dict, test_dict):
    # evaluate accuracy
    train_acc = accuracy_score(y_train_true, y_train_pred)
    test_acc = accuracy_score(y_test_true, y_test_pred)
    train_dict['acc'].append(train_acc)
    test_dict['acc'].append(test_acc)

    # evaluate f1-score
    train_f1 = f1_score(y_train_true, y_train_pred)
    test_f1 = f1_score(y_test_true, y_test_pred)
    train_dict['f1'].append(train_acc)
    test_dict['f1'].append(test_acc)

    # evaluate pre-rec-auc
    train_f1 = f1_score(y_train_true, y_train_pred)
    test_f1 = f1_score(y_test_true, y_test_pred)


def bootstrap_model(model, imgsize, test_size, mode,
                    bootstrap_epochs=20, train_epochs=20):
    x_data, y_data = load_norm_center_data(imgsize, mode)

    train, test = defaultdict(list), defaultdict(list)
    progress_bar = tqdm(range(bootstrap_epochs), smoothing=.1)
    for i in progress_bar:
        progress_bar.set_postfix(**{'avg loss': np.mean(test['loss']),
                                    'avg f1': np.mean(test['f1'])})

        # -------- BOOTSTRAPPING -------- #
        seed = dt.datetime.now().microsecond

        x_train, x_test, \
        y_train, y_test = train_test_split(x_data, y_data,
                                           stratify=y_data,
                                           test_size=test_size,
                                           random_state=seed)

        # ------------ METRICS ----------- #
        y_train_prob, y_train_pred,\
        y_test_prob, y_test_pred = evaluate_model(model,
                                             x_train, y_train,
                                             x_test, y_test,
                                             n_epochs=train_epochs)

        # evaluate loss:
        train_loss = log_loss(y_train, y_train_prob)
        test_loss = log_loss(y_test, y_test_prob)

        get_metrics(y_train_true=y_train, y_train_pred=y_train_pred,
                    y_test_true=y_test, y_test_pred=y_train_pred,
                    train_dict=train, test_dict=test)

        train['loss'].append(train_loss)
        test['loss'].append(test_loss)

    results = {
        # use uncertainties library
        'train': {
            'avg_loss': np.mean(train['loss']),
            'std_loss': np.std(train['loss']),
            'avg_acc': np.mean(train['acc']),
            'std_acc': np.std(train['acc']),
            'avg_f1': np.mean(train['f1']),
            'std_f1': np.std(train['f1']),
        },
        'test': {
            'avg_loss': np.mean(test['loss']),
            'std_loss': np.std(test['loss']),
            'avg_acc': np.mean(test['acc']),
            'std_acc': np.std(test['acc']),
            'avg_f1': np.mean(test['f1']),
            'std_f1': np.std(test['f1']),
        }
    }

    return results
