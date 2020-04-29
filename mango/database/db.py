from decouple import config
from pandas import DataFrame
from pymongo import MongoClient


class Database:
    DATABASE_NAME = 'mango-mongodb'
    COLLECTION_NAMES = ['mango-train', 'mango-cv']  # table name

    def __init__(self, user: str, password: str, cluster: str):
        self.user = user
        self.password = password
        self.cluster = cluster

        self.client = MongoClient(self.connection_string)
        self.database = self.client[self.DATABASE_NAME]

    @property
    def connection_string(self):
        return f"mongodb+srv://{self.user}:{self.password}@{self.cluster}/test?retryWrites=true&w=majority"

    def insert(self, collection, document):
        if collection in self.COLLECTION_NAMES:
            return self.database[collection].insert_one(document).inserted_id
        else:
            raise ValueError(f"{collection} is not a valid collection name. Update Database class or try a different one.")

    def list(self, collection):
        # db.list_collection_names()
        if collection in self.COLLECTION_NAMES:
            return list(self.database[collection].find())
        else:
            raise ValueError(f"{collection} is not a valid collection name. Update Database class or try a different one.")


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    user, psw, clst = config("MONGO_USER"), config("MONGO_PASSWORD"), config("MONGO_CLUSTER")
    db = Database(user, psw, clst)

    df = pd.DataFrame(db.list('mango-train')).set_index('_id')
    df = df[(df['n_classes'] == 2) & (df['model'].isin(('LeNet5', 'DenseNet'))) & (df['validation_size'] == .2)]

    lenet = df[df['model'] == 'LeNet5'].iloc[0]
    denset = df[df['model'] == 'DenseNet'].iloc[-1]

    pd.DataFrame(lenet['result']).drop(['f1_score', 'val_f1_score'], axis=1).plot()
    pd.DataFrame(denset['result']).drop(['f1_score', 'val_f1_score'], axis=1).plot()
    plt.show()








