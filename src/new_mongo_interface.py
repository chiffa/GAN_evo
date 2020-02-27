import os
from pymongo import MongoClient
import pprint


root_password = os.environ['MONGOROOTPASS']
print(root_password)
client = MongoClient(username='root', password=root_password)


gan_pair_db = client['gen-disc']
gan_train_trace = gan_pair_db['pure_train_trace']
gan_match_trace = gan_pair_db['pure_match_trace']
pure_gen_collection = gan_pair_db['pure_generator_instance']
pure_disc_collection = gan_pair_db['pure_discriminator_instance']


def save_pure_gen(payload):
    gen_id = pure_gen_collection.insert_one(payload)
    return gen_id


def save_pure_disc(payload):
    disc_id = pure_disc_collection.insert_one(payload)
    return disc_id


def filter_pure_gen(filter):

    for payload in pure_gen_collection.find(filter):
        yield payload


def filter_pure_disc(filter):

    for payload in pure_disc_collection.find(filter):
        yield payload


def purge_pure_db(match_filter={}, train_filter={}, gen_filter={}, disc_filter={}):
    r1m = gan_match_trace.delete_many(match_filter)
    r1t = gan_train_trace.delete_many(train_filter)
    r1g = pure_gen_collection.delete_many(gen_filter)
    r1d = pure_disc_collection.delete_many(disc_filter)

    print("deletion results: train/match/gen/disc: %d/%d/%d/%d" % (
        r1t.deleted_count, r1m.deleted_count, r1g.deleted_count, r1d.deleted_count))


if __name__ == "__main__":

    purge_pure_db()

    for item in filter_pure_gen({}):
        pprint.pprint(item['random_tag'], item['gen_type'])
    pass

    for item in filter_pure_disc({}):
        pprint.pprint(item['random_tag'], item['disc_type'])
    pass


