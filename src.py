import os
import glob
import numbers
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from enum import Enum

df = pd.read_parquet('data/mango_spectra.parquet')


def spectra_data_augmentation(arr, n: int, *args, **kwargs):
    l = arr.shape[1]  # graph length

    x = np.linspace(0, 1, num=l, endpoint=True)
    new_x = np.linspace(0, 1, num=l * n, endpoint=True)

    def interpolate(y):
        f = interp1d(x, y, *args, **kwargs)
        return f(new_x)
    
    result = []
    for graph in arr:
        new_graph = interpolate(graph)

        splits = defaultdict(list)
        for i, v in enumerate(new_graph):
            splits[i % n].append(v)

        result.append(
            pd.DataFrame(splits)
        )

    return pd.concat(result, axis=1).T.reset_index(drop=True)

arr = df.drop(['antrac', 'collapse'], axis=1).values

new_arr = spectra_data_augmentation(arr, n=10)
