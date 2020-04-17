import os
from functools import lru_cache

from PIL import Image
import numpy as np
from dask import delayed
from dask.array import from_delayed, stack
from tqdm import tqdm

from .functions import get_class_from_path


class IO:

    def __init__(self, root, excluded_dirs):
        self.root = root
        self.excluded = excluded_dirs

    def gen(self, **kwargs):
        """A generator to simplify dealing with the data.

        :param root: root path
        :param exclude_dirs: dirs to ignore when reading the other dirs in the root
        :param kwargs: consider_collapse_level, consider_antracnosis, max_class
        :return: (tuple) absolute image path, image class for training/validating/testing
        """
        root, exclude_dirs = self.root, self.excluded
        for dir_name in os.listdir(root):
            root_dir = os.path.join(root, dir_name)
            if os.path.isdir(root_dir) and dir_name not in exclude_dirs:
                files = os.listdir(root_dir)
                for file in files:
                    filepath = os.path.join(root_dir, file)
                    fileclass = get_class_from_path(filepath, **kwargs)

                    yield os.path.abspath(filepath), fileclass

    @staticmethod
    @lru_cache(maxsize=1024)
    def open_image(path, resize: float = 1):
        img = Image.open(path)

        if resize == 1:
            return img
        elif 0 < resize < 1:
            new_size = (np.array(img.size) * resize).astype(np.int64)
            return img.resize(new_size, Image.ANTIALIAS)
        elif resize > 1:
            new_size = [int(resize), int(resize)]
            return img.resize(new_size, Image.ANTIALIAS)
        else:
            raise ValueError(f"{resize} is not a valid resize factor")

    def open_image_array(self, *args, **kwargs):
        return np.array(self.open_image(*args, **kwargs))

    @lru_cache(maxsize=1024)
    def image(self, n, *args, **kwargs):
        generator = self.gen()

        for _ in range(n - 1):
            next(generator)

        imgpath, _ = next(generator)

        return self.open_image(imgpath, *args, **kwargs)

    def load(self, resize=1, parallel=False, *args, **kwargs):
        dataarr = np.array(list(self.gen(*args, **kwargs)))

        if parallel:
            lazy_stack = stack([
                from_delayed(
                    delayed(self.open_image_array)(path, resize), shape=self.shape, dtype=self.dtype)
                for path, _ in dataarr
            ])
            x_data = lazy_stack.compute()
        else:
            raise NotImplementedError
            X_path_arr = dataarr[..., 0]
            gen, count = (self.open_image(path, resize)
                          for path in tqdm(X_path_arr, ncols=80)), X_path_arr.shape[0]

            x_data = np.empty(shape=(count, self.shape), dtype=np.uint8)
            for i, img in enumerate(gen):
                x_data[i] = img

        try:
            y_data = dataarr[:, :, 1].astype(
                np.uint8).mean(axis=1).astype(np.uint8)
        except IndexError:
            y_data = dataarr[:, 1].astype(np.uint8)

        return x_data, y_data

    @property
    def shape(self):
        return np.array(self.image(0)).shape

    @property
    def dtype(self):
        return np.array(self.image(0)).dtype
