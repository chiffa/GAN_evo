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

    gen_id = gen_collection.insert_one({'weights': pickle.dumps(gen_dump)})
    disc_id = disc_collection.insert_one({'weights': pickle.dumps(disc_dump)})

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

    return gan_trainer_collection.find(filter_dict)  # that returns a cursor - aka an iterator


def gan_pair_purge_db():

    r1 = client['gan-disc']['main'].delete_many({})
    r2 = client['image']['main'].delte_many({})

    print("deletion results: gan/disc: %s; images: %s" % (r1.deleted_count,
                                                          r2.deleted_count))


if __name__ == "__main__":
    gans_db = client['gan-disc']
    gans_db_main = gans_db['main']
    gans_db_gen = gans_db['generator']
    gans_db_disc = gans_db['discriminator']

    image_db = client['image']
    image_db_main = image_db['main']

    gan_pair_purge_db()


