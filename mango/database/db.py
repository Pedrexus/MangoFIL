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

    user, psw, clst = config("MONGO_USER"), config("MONGO_PASSWORD"), config("MONGO_CLUSTER")
    db = Database(user, psw, clst)

    df = DataFrame(db.list('mango-cv')).set_index('_id')

    df = df[(df['n_classes'] == 2) & (df['data_size'] > 100)]
    df = df.drop(['n_classes', 'input_shape', 'data_size', 'model_kwargs', 'loss', 'optimizer', 'test_size', 'validation_size', 'random_state', 'augmentation'], axis=1)

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






