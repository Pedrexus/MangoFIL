def print_results(model, training_data=None,
                  validation_data=None,
                  test_data=None):
    if training_data:
        train_loss, train_acc, *_ = model.evaluate(training_data[0],
                                               training_data[1],
                                               verbose=0)
        train_str = f'Train loss: {train_loss:.4f}\nTrain accuracy: {train_acc:.4f}'
        print('-' * 32, train_str, sep='\n')

    if validation_data:
        valid_loss, valid_acc, *_ = model.evaluate(validation_data[0],
                                               validation_data[1],
                                               verbose=0)
        valid_str = f'Valid loss: {valid_loss:.4f}\nValid accuracy: {valid_acc:.4f}'
        print('-' * 32, valid_str, sep='\n')

    if test_data:
        test_loss, test_acc, *_ = model.evaluate(test_data[0], test_data[1],
                                             verbose=0)
        test_str = f'Test loss: {test_loss:.4f}\nTest accuracy: {test_acc:.4f}'
        print('-' * 32, test_str, sep='\n')

    print('-' * 32)


def print_avg_results_dict(results_dict):
    _r = results_dict

    train_str = f'Train loss: {_r["train"]["avg_loss"]:.4f}' \
                f' +- {_r["train"]["std_loss"]:.4f}\n' \
                f'Train accuracy: {_r["train"]["avg_acc"]:.4f}' \
                f' +- {_r["train"]["std_acc"]:.4f}'

    print('-' * 32, train_str, sep='\n')

    test_str = f'Test loss: {_r["test"]["avg_loss"]:.4f}' \
               f' +- {_r["test"]["std_loss"]:.4f}\n' \
               f'Test accuracy: {_r["test"]["avg_acc"]:.4f}' \
               f' +- {_r["test"]["std_acc"]:.4f}'

    print('-' * 32, test_str, sep='\n')