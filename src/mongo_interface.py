import os
from pymongo import MongoClient
import pprint
import pickle
from bson.objectid import ObjectId


root_password = os.environ['MONGOROOTPASS']
print(root_password)
client = MongoClient(username='root', password=root_password)

gan_pair_db = client['gen-disc']
gan_trainer_collection = gan_pair_db['trainer']
gen_collection = gan_pair_db['generator']
disc_collection = gan_pair_db['discriminator']


def gan_pair_push_to_db(payload):

    def check_payload():
        with open("mongo_debug_dump.txt", "w") as fout:
            fout.write(pprint.pformat(payload))

    check_payload()

    gen_dump = payload['Generator_state']
    disc_dump = payload['Discriminator_state']

    gen_id = gen_collection.insert_one({'weights': pickle.dumps(gen_dump)}).inserted_id
    disc_id = disc_collection.insert_one({'weights': pickle.dumps(disc_dump)}).inserted_id

    payload['Generator_state'] = str(gen_id)
    payload['Discriminator_state'] = str(disc_id)

    insertion_id = gan_trainer_collection.insert_one(payload)

    return insertion_id


def gan_pair_get_from_db(key):

    payload = gan_trainer_collection.find_one(key)

    if payload is not None:
        gen_id = payload['Generator_state']
        disc_id = payload['Discriminator_state']
        gen_dump = pickle.loads(gen_collection.find_one({"_id": ObjectId(gen_id)})['weights'])
        disc_dump = pickle.loads(disc_collection.find_one({"_id": ObjectId(disc_id)})['weights'])
        payload['Generator_state'] = gen_dump
        payload['Discriminator_state'] = disc_dump

    else:
        return None


def gan_pair_update_in_db(key, update_payload):

    if 'Generator_state' in update_payload.keys() or 'Discriminator_state' in update_payload.keys():
        existing_trainer_payload = gan_trainer_collection.find_one(key)
        gen_id = existing_trainer_payload['Generator_state']
        disc_id = existing_trainer_payload['Discriminator_state']

        if 'Generator_state' in update_payload.keys():
            gen_dump = update_payload['Generator_state']
            gen_collection.find_one_and_update({"_id": ObjectId(gen_id)},
                                               {"$set": {'weights': pickle.dumps(gen_dump)}})
            del update_payload['Generator_state']

        if 'Discriminator_state' in update_payload.keys():
            disc_dump = update_payload['Discriminator_state']
            disc_collection.find_one_and_update({"_id": ObjectId(disc_id)},
                                                {"$set": {'weights': pickle.dumps(disc_dump)}})
            del update_payload['Discriminator_state']

    update_result = gan_trainer_collection.find_one_and_update(key,
                                                               {"$set": update_payload})

    return update_result


def gan_pair_list_by_filter(filter_dict):

    for payload in gan_trainer_collection.find(filter_dict):
        gen_id = payload['Generator_state']
        disc_id = payload['Discriminator_state']
        gen_dump = pickle.loads(gen_collection.find_one({"_id": ObjectId(gen_id)})['weights'])
        disc_dump = pickle.loads(disc_collection.find_one({"_id": ObjectId(disc_id)})['weights'])
        payload['Generator_state'] = gen_dump
        payload['Discriminator_state'] = disc_dump

        yield payload


def gan_pair_purge_db():

    r1 = gan_trainer_collection.delete_many({})
    r1d = disc_collection.delete_many({})
    r1g = gen_collection.delete_many({})
    # r2 = client['image']['main'].delte_many({})

    print("deletion results: train/gen/disc: %d/%d/%d" % (
        r1.deleted_count, r1g.deleted_count, r1d.deleted_count))


def gan_pair_eliminate(filter_dict):

    for payload in gan_trainer_collection.find(filter_dict):
        gen_id = payload['Generator_state']
        disc_id = payload['Discriminator_state']
        gen_collection.delete_one({"_id": ObjectId(gen_id)})
        disc_collection.delete_one({"_id": ObjectId(disc_id)})
        gan_trainer_collection.delete_one({"_id": payload['_id']})



if __name__ == "__main__":

    # gan_pair_purge_db()

    for item in gan_pair_list_by_filter({}):
        pprint.pprint(item['random_tag'])
    pass


