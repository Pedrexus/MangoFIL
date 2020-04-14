import os
import numpy as np
from sklearn.model_selection import train_test_split


def fetch_data(imgsize, mode='A', test_size=.1, seed=12345):
    x_data, y_data = load_norm_center_data(imgsize, mode)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        stratify=y_data,
                                                        test_size=test_size,
                                                        random_state=seed)

    # y_train = np_utils.to_categorical(y_train, n_classes)
    # y_test = np_utils.to_categorical(y_test, n_classes)

    return x_train, y_train, x_test, y_test


def load_norm_center_data(imgsize, mode):
    root = r'D:\Coding\Python\MangoFIL\data\NPZ_FILES'
    filename = f'data_{imgsize!r}px.npz'
    if mode is 'A':
        dirc = 'ANTRAC_2'
        n_classes = 2
    elif mode is 'C':
        dirc = 'COLLAPSE_2'
        n_classes = 2
    elif mode is 'AC':
        dirc = 'ANTRAC_COLLAPSE_4'
        n_classes = 4
    else:
        raise NotImplementedError
    with np.load(os.path.join(root, dirc, filename)) as f:
        x_data, y_data = f['x'], f['y']
    x_data = x_data.astype('float32')
    x_data /= 255

    mean_vals = np.mean(x_data, axis=0)
    std_val = np.std(x_data)
    x_data = (x_data - mean_vals) / std_val

    return x_data, y_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from collections import Counter

    cls_dict = {0: 'sem Colapso e sem Antracnose',
                1: 'apenas Colapso',
                2: 'apenas Antracnose',
                3: 'Colapso e Antracnose'}

    x_data, y_data = load_norm_center_data(32, 'AC')
    y_hist = [cls_dict[y] for y in y_data]
    c = Counter(y_hist)

    import pandas as pd
    from collections import Counter

    df = pd.DataFrame.from_dict(c, orient='index')
    df.plot(kind='bar')