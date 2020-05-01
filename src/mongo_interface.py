"""
Collection of the methods to connect to the MogoDB for the storage of the generator/discriminator states.
"""
import os
from pymongo import MongoClient
import pprint
from bson.objectid import ObjectId


root_password = os.environ['MONGOROOTPASS']
client = MongoClient(username='root', password=root_password)


gan_pair_db = client['gen-disc']
gan_trace = gan_pair_db['pure_trace']
gan_match_trace = gan_pair_db['pure_match_trace']
pure_gen_collection = gan_pair_db['pure_generator_instance']
pure_disc_collection = gan_pair_db['pure_discriminator_instance']


def separate_trace_save(trace):
    trace_id = gan_trace.insert_one({'load': trace}).inserted_id
    return str(trace_id)


def separate_trace_retrieve(trace_id):
    trace = gan_trace.find_one({"_id": ObjectId(trace_id)})['load']
    return trace


def separate_trace_update(trace_id, new_trace):
    gan_trace.find_one_and_update({"_id": ObjectId(trace_id)},
                                  {'$set': {'load':new_trace}})
    return trace_id


def save_gen(payload):
    payload['encounter_trace'] = separate_trace_save(payload['encounter_trace'])
    gen_id = pure_gen_collection.insert_one(payload).inserted_id
    return gen_id


def save_disc(payload):
    payload['encounter_trace'] = separate_trace_save(payload['encounter_trace'])
    disc_id = pure_disc_collection.insert_one(payload).inserted_id
    return disc_id


def update_gen(key, update_payload):
    existing_gen = pure_gen_collection.find_one({'random_tag': key})
    trace_id = existing_gen['encounter_trace']
    update_payload['encounter_trace'] = separate_trace_update(trace_id,
                                                              update_payload['encounter_trace'])
    update_result = pure_gen_collection.find_one_and_update({'random_tag': key},
                                                            {'$set': update_payload})
    return update_result


def update_disc(key, update_payload):
    existing_disc = pure_disc_collection.find_one({'random_tag': key})
    trace_id = existing_disc['encounter_trace']
    update_payload['encounter_trace'] = separate_trace_update(trace_id,
                                                              update_payload['encounter_trace'])

    update_result = pure_disc_collection.find_one_and_update({'random_tag': key},
                                                             {'$set': update_payload})
    return update_result


def gen_from_random_tag(random_tag):
    existing_gen = pure_gen_collection.find_one({'random_tag': random_tag})
    existing_gen['encounter_trace'] = separate_trace_retrieve(existing_gen['encounter_trace'])
    return existing_gen


def disc_from_random_tag(random_tag):
    existing_disc = pure_disc_collection.find_one({'random_tag': random_tag})
    existing_disc['encounter_trace'] = separate_trace_retrieve(existing_disc['encounter_trace'])
    return existing_disc


def filter_gens(filter):

    for payload in pure_gen_collection.find(filter):
        payload['encounter_trace'] = separate_trace_retrieve(payload['encounter_trace'])
        yield payload


def filter_discs(filter):

    for payload in pure_disc_collection.find(filter):
        payload['encounter_trace'] = separate_trace_retrieve(payload['encounter_trace'])
        yield payload


def purge_db(train_filter={}, gen_filter={}, disc_filter={}):
    r1t = gan_trace.delete_many(train_filter)
    r1g = pure_gen_collection.delete_many(gen_filter)
    r1d = pure_disc_collection.delete_many(disc_filter)

    print("deletion results: train/match/gen/disc: %d/%d/%d/%d" % (
        r1t.deleted_count, -1, r1g.deleted_count, r1d.deleted_count))


if __name__ == "__main__":

    for item in filter_gens({}):
        print(item['random_tag'], item['gen_type'])
    pass

    for item in filter_discs({}):
        print(item['random_tag'], item['disc_type'])
    pass


