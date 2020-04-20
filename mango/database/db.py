import datetime as dt

from pymongo import MongoClient


class Database:

    DATABASE_NAME = 'mango-mongodb'
    COLLECTION_NAME = 'mango-train'  # table name

    def __init__(self, user: str, password: str, cluster: str):
        self.user = user
        self.password = password
        self.cluster = cluster

        self.client = MongoClient(self.connection_string)
        self.collection = self.client[self.DATABASE_NAME][self.COLLECTION_NAME]

    @property
    def connection_string(self):
        return f"mongodb+srv://{self.user}:{self.password}@{self.cluster}/test?retryWrites=true&w=majority"

    def insert(self, document):
        return self.collection.insert_one(document).inserted_id

    def list(self):
        # db.list_collection_names()
        return list(self.collection.find())


