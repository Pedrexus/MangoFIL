import mango

from numpy import uint8
from PIL.Image import Image

def create_io():
    root = 'MangoFIL/data'
    exclude_dirs = ('FIL_NANDA', 'EXCLUDED', 'NPZ_FILES')

    io = mango.IO(root, exclude_dirs)

    return io

def test_create_image_data_generator(benchmark):
    io = create_io()

    imggen = io.gen(consider_antracnosis='only', consider_collapse_level=False)
    imggen = benchmark(list, imggen)
    _path, _class = next(imggen)

    assert type(_path) == str
    assert type(_class) == int
    assert set(x[1] for x in imggen) == {0, 1}

def test_get_image_characteristics():
    io = create_io()

    assert io.shape == (2760, 4912, 3)
    assert io.dtype == uint8

def test_visualize_resized_image():
    io = create_io()

    img = io.image(10, resize=.1)

    assert type(img) == Image
    assert img.size == (491, 276)

    img = io.image(10, resize=32)

    assert type(img) == Image
    assert img.size == (32, 32)

def test_load_image_data_parallel(benchmark):
    io = create_io()

    kwargs = dict(consider_antracnosis='only', consider_collapse_level=False)
    x_data, y_data = benchmark(io.load, resize=.1, parallel=True, **kwargs)

    assert x_data.shape == (470, 491, 276, 3)
    assert x_data.dtype == uint8

    assert y_data.shape == (470,)
    assert y_data.dtype == uint8

def test_load_image_data_single_thread(benchmark):
    io = create_io()

    kwargs = dict(consider_antracnosis='only', consider_collapse_level=False)
    x_data, y_data = benchmark(io.load, resize=28, parallel=False)

    assert x_data.shape == (470, 28, 28, 3)
    assert x_data.dtype == uint8

    assert y_data.shape == (470,)
    assert y_data.dtype == uint8
    