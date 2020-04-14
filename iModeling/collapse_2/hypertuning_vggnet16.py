from iModeling.functions import show_and_save_best_model
from iModeling.imports import *
from iModeling.models import create_model_vggnet16


def data():
    IMGSIZE = 256
    x_train, y_train, x_test, y_test = fetch_data(IMGSIZE,
                                                  mode='C',
                                                  test_size=.3,
                                                  seed=123)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model_vggnet16,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = data()

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    show_and_save_best_model(best_model,
                             *data(),
                             dataset_name='Collapse_2',
                             architecture_name='VGGNet-16',
                             notes='acc=F1, imgsize=64px')
