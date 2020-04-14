from iModeling.functions import show_and_save_best_model
from iModeling.imports import *
from iModeling.models import create_model_alexnet

# activate GPUs
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def data():
    IMGSIZE = 256
    x_train, y_train, x_test, y_test = fetch_data(IMGSIZE,
                                                  mode='AC',
                                                  test_size=.3,
                                                  seed=123)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model_alexnet,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = data()

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    show_and_save_best_model(best_model,
                             *data(),
                             dataset_name='Antrac_Collapse_4',
                             architecture_name='AlexNet',
                             notes='acc=F1, imgsize=256px')