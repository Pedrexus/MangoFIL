from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from scipy.interpolate import interp1d


def one_hot_encode(arr):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(arr)
    return enc.transform(arr).toarray()

def spectra_data_augmentation(arr, labels, n: int, *args, **kwargs):
    l = arr.shape[1]  # graph length

    x = np.linspace(0, 1, num=l, endpoint=True)
    new_x = np.linspace(0, 1, num=l * n, endpoint=True)

    def interpolate(y):
        f = interp1d(x, y, *args, **kwargs)
        return f(new_x)
    
    result = []
    relabeled = []
    for i, graph in enumerate(arr):
        new_graph = interpolate(graph)

        splits = defaultdict(list)
        for j, v in enumerate(new_graph):
            splits[j % n].append(v)

        result.append(
            pd.DataFrame(splits)
        )
        
        label = [labels[i]] * n
        relabeled = [*relabeled, *label]

    return pd.concat(result, axis=1).T.reset_index(drop=True), np.array(relabeled)

