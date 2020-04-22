import datetime as dt

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


