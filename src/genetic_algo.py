import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from src.mongo_interface import gan_pair_eliminate, gan_pair_list_by_filter
from src.gans.gan_disc import GanTrainer
from src.arena import run_match
from copy import deepcopy

#TODO: factor this out to a single modification point
image_folder = "./image"
image_size = 64
number_of_colors = 1
imtype = 'mnist'

mnist_dataset = dset.MNIST(root=image_folder, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))


immutables = {'dataset': mnist_dataset,
              'number_of_colors': number_of_colors,
              'image_dimensions': image_size,
              'image_type': 'mnist'}

image_chars = (immutables['image_type'],
               immutables['image_dimensions'],
               type(immutables['dataset']).__name__)

mutables = {'latent_vector_size': 64,
            'generator_latent_maps': 64,
            'discriminator_latent_maps': 64,
            'learning_rate': 0.0002,
            'beta1': 0.5,
            'training_epochs': 25}


def clear_weak(destruction_bound):

    gan_pair_eliminate({'image_chars': image_chars,
                        'score_ratings.1': {'$lt': destruction_bound}})

    gan_pair_eliminate({'image_chars': image_chars,
                        'score_ratings.2': {'$lt': destruction_bound}})


def unpack_mongo_payload(mongo_payload):
    mutable_payload = {}

    mutable_payload['latent_vector_size'], \
    mutable_payload['discriminator_latent_maps'],\
    mutable_payload['generator_latent_maps'] = \
        mongo_payload['latent_maps_params']
    mutable_payload['learning_rate'], \
    mutable_payload['beta1'], \
    mutable_payload['training_epochs'], _, _, _, _, _ = \
        mongo_payload['training_parameters']
    mutable_payload['score_ratings'] = mongo_payload['score_ratings']

    return mutable_payload

# def repack_mongo_payload(mutable_payload, mongo_payload):
#     mongo_payload['latent_maps_params'] = (mutable_payload['batch_size'],
#                                            mutable_payload['latent_vector_size'],
#                                            mutable_payload['generator_latent_maps'])
#
#     mongo_payload['training_parameters'] = (mutable_payload['learning_rate'],
#                                              mutable_payload['beta1'],
#                                              mutable_payload['training_epochs'],
#                                             mongo_payload['training_parameters'][3],
#                                             mongo_payload['training_parameters'][4],
#                                             mongo_payload['training_parameters'][5],
#                                             mongo_payload['training_parameters'][6],
#                                             mongo_payload['training_parameters'][7])


def reproduce(genetic_pool_size):
    base_dicts = [unpack_mongo_payload(payload) for payload
                  in gan_pair_list_by_filter({'image_chars': image_chars})]

    for dict in base_dicts:
        print(dict.keys())
        print(dict['score_ratings'])

    probability_distro = np.array([payload['score_ratings'][1]+payload['score_ratings'][1] for \
            payload in base_dicts])
    probability_distro /= np.sum(probability_distro)

    draw = np.random.choice(range(len(base_dicts)), genetic_pool_size, p=probability_distro)

    for dict in base_dicts:
        del dict['score_ratings']

    return [deepcopy(base_dicts[elt_num]) for elt_num in draw]


def mutate(mutable_dict, mutation_intensity):
    for key, value in mutables.items():
        print('mutating %s from %s' % (key, mutable_dict[key]), end='')
        mutable_dict[key] += np.random.normal(0, mutation_intensity*value)
        mutable_dict[key] = type(value)(mutable_dict[key])
        if mutable_dict[key] < 0:
            mutable_dict[key] = - mutable_dict[key]
        print(' to %s' % mutable_dict[key])


def generation_round(weak_clear_bound, mutation_intensity, genetic_pool_size):

    clear_weak(destruction_bound=weak_clear_bound)
    new_gen_params = reproduce(genetic_pool_size)

    new_gen = []
    for trainer_params in new_gen_params:
        mutate(trainer_params, mutation_intensity)
        trainer_params.update(immutables)
        new_gen.append(GanTrainer(**trainer_params))

    return new_gen


if __name__ == "__main__":
    run_match(mnist_dataset)
    for _ in range(0, 5):
        new_gen = generation_round(1100, 0.05, 4)
        for trainer in new_gen:
            trainer.do_pair_training()
            trainer.save()
        run_match(mnist_dataset)


