import os

def get_class_from_path(filepath, consider_collapse_level=True, consider_antracnosis=True, max_class=3):
    name = os.path.basename(filepath)
    collapse_level = name.split('_')[2][1]

    try:
        collapse_level = int(collapse_level)
        collapse_level = collapse_level if consider_collapse_level else 1
    except ValueError:
        # S = C0
        collapse_level = 0

    if consider_antracnosis and 'antracnose' in os.path.dirname(filepath):
        # with antracnosis, classes are 4, 5, 6, 7
        if consider_antracnosis is 'only':
            return 1

        return collapse_level + (max_class + 1)

    # without antracnosis, classes are 0, 1, 2, 3
    if consider_antracnosis is 'only':
        return 0

    return collapse_level