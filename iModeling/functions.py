import os

import pandas as pd
from keras.engine.saving import load_model

ROOT = r'D:\Coding\Python\MangoFIL\iModeling'
FILENAME = 'results.json'
BACKUPNAME = 'results_backup.json'

FILEPATH = os.path.join(ROOT, FILENAME)
BACKUPPATH = os.path.join(ROOT, BACKUPNAME)

SAVEPATH = r'D:\Coding\Python\MangoFIL\iModeling\saved_models'


def save_results(best_model, x_test, y_test,
                 dataset_name, architecture_name, notes):
    df = pd.read_json(FILEPATH)
    data = df[(df['dataset'] == dataset_name) &
              (df['arch'] == architecture_name)]

    new_loss, new_acc, *_ = best_model.evaluate(x_test, y_test, verbose=0)
    old_acc = data['acc'].iloc[0] if any(data['acc'].values) else 0

    if new_acc > old_acc:
        save_name = f'{dataset_name}_{architecture_name}.h5'
        save_path = os.path.join(SAVEPATH, save_name)
        best_model.save(save_path, overwrite=True, include_optimizer=True)

        new_data = [dataset_name, architecture_name, new_loss, new_acc, notes]
        try:
            idx = data.index.values[0]
            df.loc[idx] = new_data
        except IndexError:
            data.loc[0] = new_data
            df = df.append(data, ignore_index=True)

        df.to_json(FILEPATH, force_ascii=False)
        print('Saving completed')

    else:
        backup_results()
        print('Model not saved. Backing up instead.')


def backup_results():
    df = pd.read_json(FILEPATH)
    df.to_json(BACKUPPATH)


def create_results_file():
    df = pd.DataFrame(columns=['dataset', 'arch', 'loss', 'acc', 'notes'])
    df.to_json(FILEPATH)


def show_and_save_best_model(best_model,
                             x_train, y_train, x_test, y_test,
                             dataset_name, architecture_name, notes):

    print("Results of best performing model:")
    from iNeural.printing import print_results
    print_results(best_model,
                  training_data=(x_train, y_train),
                  test_data=(x_test, y_test))

    print("Saving if model is better:")
    save_results(best_model, x_test, y_test,
                 dataset_name, architecture_name, notes)


def model_results(model_path, max_epochs, x_train, y_train):
    best_model = load_model(model_path)
    reset_weights(best_model)

    from sklearn.model_selection import train_test_split
    x_tr, x_valid, y_tr, y_valid = train_test_split(x_train, y_train,
                                                    stratify=y_train,
                                                    test_size=.25,
                                                    random_state=123)
    result = best_model.fit(x_tr, y_tr,
                            batch_size=16,
                            epochs=max_epochs,
                            verbose=0,
                            validation_data=(x_valid, y_valid),
                            shuffle=True)
    return result


if __name__ == '__main__':
    create_results_file()
