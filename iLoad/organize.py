import os

from iLoad.classification import get_class_from_path


def images_gen(root, exclude_dirs, **kwargs):
    """A generator to simplify dealing with the data.

    :param root: root path
    :param exclude_dirs: dirs to ignore when reading the other dirs in the root
    :param kwargs: consider_collapse_level, consider_antracnosis, max_class
    :return: (tuple) absolute image path, image class for
                     training/validating/testing
    """
    for dir_name in os.listdir(root):
        root_dir = os.path.join(root, dir_name)
        if os.path.isdir(root_dir) and dir_name not in exclude_dirs:
            files = os.listdir(root_dir)
            for file in files:
                filepath = os.path.join(root_dir, file)
                fileclass = get_class_from_path(filepath, **kwargs)

                yield os.path.abspath(filepath), fileclass


def list_data(n_classes):
    root, exclude_dirs = r'D:\Coding\Python\MangoFIL\data', ('FIL_NANDA',
                                                             'EXCLUDED',
                                                             'NPZ_FILES')

    if n_classes is 2:
        # classes are: with colapse, without colapse - COLLAPSE_2
        kwargs = dict(consider_antracnosis=False,
                      consider_collapse_level=False)
    elif n_classes is '2':
        # ANTRAC_2
        kwargs = dict(consider_antracnosis='only',
                      consider_collapse_level=False)
    elif n_classes is 4:
        # classes are: (colapse, no colapse) X (antrac, no antrac)
        kwargs = dict(consider_antracnosis=True,
                      consider_collapse_level=False,
                      max_class=1)
    elif n_classes is '4':
        # classes are: (no colapse, clp1, clp2, clp3)
        kwargs = dict(consider_antracnosis=False,
                      consider_collapse_level=True)
    elif n_classes is 8:
        # classes are: (no colapse, clp1, clp2, clp3) X (antrac, no antrac)
        kwargs = dict(consider_antracnosis=True,
                      consider_collapse_level=True)
    else:
        raise NotImplementedError

    imggen = images_gen(root, exclude_dirs, **kwargs)

    return list(imggen)


if __name__ == '__main__':
    from collections import Counter
    import numpy as np

    root = r'..\data'
    exclude_dirs = ('FIL_NANDA', 'EXCLUDED', 'NPZ_FILES')
    imggen = images_gen(root, exclude_dirs,
                        consider_antracnosis=False,
                        consider_collapse_level=False)

    class_set = set(x[1] for x in imggen)
    print(class_set)

    ns = [2, '2', 4, '4', 8]
    classes = {n: Counter(np.array(list_data(n))[:, 1]) for n in ns}
