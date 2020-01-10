import os
from pymongo import MongoClient


root_password = os.environ('MONGOROOTPASS')
client = MongoClient(user='root', password=root_password)


def push_to_db(payload, type):
    # payload: {md}

    def check_payload():
        pass

    if type not in ['gan-disc', 'image']:
        raise(Exception("%s is not a supported collection in the database" % type))

    collection = client[type]

    insertion_id = collection.insert_one(payload).inserted_id

    return insertion_id


def get_from_db(key, type):

    if type not in ['gan-disc', 'image']:
        raise(Exception("%s is not a supported collection in the database" % type))

    collection = client[type]

    return collection.find_one(key)


if __name__ == "__main__":
    gans_db = client['gan-disc']
    image_db = client['image']
