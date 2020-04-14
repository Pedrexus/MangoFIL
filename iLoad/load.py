import numpy as np
from tqdm import tqdm
from PIL import Image

from iLoad.organize import list_data


def load_set(path_dataset, resize: float, imgshape):

    try:
        X_path_arr = path_dataset[:, :, 0]
    except IndexError:
        X_path_arr = path_dataset[:, 0]

    try:
        y_data = path_dataset[:, :, 1] \
            .astype(np.uint8) \
            .mean(axis=1) \
            .astype(np.uint8)
    except IndexError:
        y_data = path_dataset[:, 1] \
            .astype(np.uint8)

    if 0 < resize < 1:
        # resize is resize_percentage
        new_shape = (int(imgshape[0] * resize), int(imgshape[1] * resize))
    else:
        # resize is pixel_width and pixel_height
        new_shape = (int(resize), int(resize))

    X_data = []

    for imgset in tqdm(X_path_arr, ncols=80):
        path_dataset = []
        try:
            for imgside in imgset:
                img = Image.open(imgside).resize(new_shape, Image.ANTIALIAS)
                path_dataset.append(np.array(img))
        except FileNotFoundError:
            imgpath = imgset
            img = Image.open(imgpath).resize(new_shape, Image.ANTIALIAS)
            path_dataset = np.array(img)

        X_data.append(path_dataset)

    X_data = np.array(X_data, dtype=np.uint8)

    return X_data, y_data


def save_data(filepath, n_classes, resize, imgshape=(2760, 4912, 3)):
    unload_data = np.array(list_data(n_classes))
    X_data, y_data = load_set(unload_data, resize, imgshape)
    np.savez(filepath, x=X_data, y=y_data)


if __name__ == '__main__':
    import os

    root = r'D:\Coding\Python\MangoFIL\data\NPZ_FILES'
    sizes = [2**n for n in range(5, 10)]
    dirs = {'ANTRAC_2': '2', 'COLLAPSE_2': 2, 'ANTRAC_COLLAPSE_4': 4}
    size = 32
    for dir, n_cls in dirs.items():
        path = os.path.join(root, dir, f'data_{size!r}px.npz')
        save_data(path, n_classes=n_cls, resize=size)
