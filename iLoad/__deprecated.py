import numpy as np

from iLoad.mfil import list_data
from iLoad.load import load_set


def load_data(n_classes, random_seed, train_pct, valid_pct, resize,
              imgshape=(2760, 4912, 3), verbose=False,
              standardize=False, reshape_by_sides=False):
    unload_data = np.array(list_data(n_classes))

    if reshape_by_sides:
        unload_data = unload_data.reshape((-1, 4, 4))

    np.random.seed(random_seed)
    np.random.shuffle(unload_data)

    train_set, valid_set, test_set = split_train_valid_test(
        unload_data, train_pct, valid_pct)

    if verbose:
        print("loading training set...")
    X_train, y_train = load_set(train_set, resize, imgshape)
    if verbose:
        print("loading validation set...")
    X_valid, y_valid = load_set(valid_set, resize, imgshape)
    if verbose:
        print("loading test set...")
    X_test, y_test = load_set(test_set, resize, imgshape)

    if verbose:
        print("normalizing arrays...")
    # TODO: allow 2 norm options
    X_train = X_train / 255
    X_valid = X_valid / 255

    if standardize:
        if verbose:
            print("calculating mean and std...")
        mean_vals = np.mean(X_train, axis=0)
        std_val = np.std(X_train)

        if verbose:
            print("centering arrays...")
        X_train_centered = (X_train - mean_vals) / std_val
        X_valid_centered = X_valid - mean_vals
        X_test_centered = (X_test - mean_vals) / std_val

        del X_train, X_valid

        training_set = (X_train_centered, y_train)
        validation_set = (X_valid_centered, y_valid)
        test_set = (X_test_centered, y_test)
    else:
        training_set = (X_train, y_train)
        validation_set = (X_valid, y_valid)
        test_set = (X_test, y_test)

    return training_set, validation_set, test_set


def split_train_valid_test(arr, train_pct, valid_pct):
    length = arr.shape[0]
    train_size = int(np.ceil(length * train_pct))
    valid_size = train_size + int(np.ceil(length * valid_pct))
    test_size = length

    return np.split(arr, [train_size, valid_size, test_size])[:-1]