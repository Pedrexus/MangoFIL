import numpy as np
import pandas as pd
from decouple import config

from mango import Database

user, psw, clst = config("MONGO_USER"), config("MONGO_PASSWORD"), config("MONGO_CLUSTER")
db = Database(user, psw, clst)

df = pd.DataFrame(db.list('mango-cv')).set_index('_id')

df = df[(df['n_classes'] == 2) & (df['data_size'] > 100)]
df = df.drop(['n_classes', 'input_shape', 'data_size', 'loss', 'optimizer', 'test_size', 'validation_size',
              'random_state'], axis=1)


def train_f1_score(row):
    r = row.result
    train_f1_scores = [x[0]["f1_score"][-1] for x in r]
    return np.mean(train_f1_scores), np.std(train_f1_scores)


def test_f1_score(row):
    r = row.result
    test_f1_scores = [x[1][1] for x in r]
    return np.mean(test_f1_scores), np.std(test_f1_scores)


df["train_f1_mean"], df["train_f1_std"] = df.apply(train_f1_score, axis=1, result_type='expand').T.values
df["test_f1_mean"], df["test_f1_std"] = df.apply(test_f1_score, axis=1, result_type='expand').T.values

lenet = df[df['model'] == 'LeNet5']
densenet = df[df['model'] == 'DenseNet']

df_with_da = df[df['augmentation'].apply(bool) & (df['n_splits'] == 5)]

lenet_with_da = df_with_da[df_with_da['model'] == 'LeNet5']
densenet_with_da = df_with_da[df_with_da['model'] == 'DenseNet']

best_lenet_with_da = lenet_with_da[lenet_with_da['test_f1_mean'] == lenet_with_da.max()['test_f1_mean']].iloc[0]
best_densenet_with_da = densenet_with_da[densenet_with_da['test_f1_mean'] == densenet_with_da.max()['test_f1_mean']].iloc[0]

#

df_without_da = df[~df['augmentation'].apply(bool) & (df['n_splits'] == 5)]

lenet_without_da = df_without_da[df_without_da['model'] == 'LeNet5']
densenet_without_da = df_without_da[df_without_da['model'] == 'DenseNet']

best_lenet_without_da = lenet_without_da[lenet_without_da['test_f1_mean'] == lenet_without_da.max()['test_f1_mean']].iloc[0]
best_densenet_without_da = \
densenet_without_da[densenet_without_da['test_f1_mean'] == densenet_without_da.max()['test_f1_mean']].iloc[0]

#

best_lenet_with_da.name = "LeNet5 (With DA)"
best_lenet_without_da.name = "LeNet5 (No DA)"

best_densenet_with_da.name = "DenseNet (With DA)"
best_densenet_without_da.name = "DenseNet (No DA)"

result = pd.concat([best_lenet_with_da, best_lenet_without_da, best_densenet_with_da, best_densenet_without_da], axis=1)
result.drop(['augmentation', 'result', 'model_kwargs'], inplace=True)
