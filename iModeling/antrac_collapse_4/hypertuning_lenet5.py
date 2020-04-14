from iModeling.functions import show_and_save_best_model
from iModeling.imports import *
from iModeling.models import create_model_lenet5

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def data():
    IMGSIZE = 32

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    x_train, y_train, x_test, y_test = fetch_data(IMGSIZE,
                                                  mode='AC',
                                                  test_size=.3,
                                                  seed=123)

    # x_train = rgb2gray(x_train).reshape((-1, IMGSIZE, IMGSIZE, 1))
    # x_test = rgb2gray(x_test).reshape((-1, IMGSIZE, IMGSIZE, 1))

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=create_model_lenet5,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=200,
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = data()

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    show_and_save_best_model(best_model,
                             *data(),
                             dataset_name='Antrac_Collapse_4',
                             architecture_name='LeNet-5',
                             notes='acc=F1, imgsize=32px')
