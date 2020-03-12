from src.gans.discriminator_zoo import Discriminator, Discriminator_PReLU, Discriminator_light
from src.gans.generator_zoo import Generator
from src.gans.trainer_zoo import Arena, GANEnvironment
# from src.new_mongo_interface import save_pure_disc, save_pure_gen, filter_pure_disc, filter_pure_gen

import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import combinations, product
import torch.optim as optim


def spawn_host_population(individuals_per_species):
    hosts = {
        'base': [],
        'PreLU': [],
        'light': []
    }  # Type: list
    for _ in range(0, individuals_per_species):
        hosts['base'].append(Discriminator(ngpu=environment.ngpu,
                         latent_vector_size=environment.latent_vector_size,
                         discriminator_latent_maps=64,
                         number_of_colors=environment.number_of_colors).to(environment.device))
        hosts['PreLU'].append(Discriminator_PReLU(ngpu=environment.ngpu,
                         latent_vector_size=environment.latent_vector_size,
                         discriminator_latent_maps=64,
                         number_of_colors=environment.number_of_colors).to(environment.device))
        hosts['light'].append(Discriminator(ngpu=environment.ngpu,
                                         latent_vector_size=environment.latent_vector_size,
                                         discriminator_latent_maps=32,
                                         number_of_colors=environment.number_of_colors).to(environment.device))

    for host_type, _hosts in hosts.items():
        print(host_type, ': ', [host.random_tag for host in _hosts])

    return hosts


def spawn_pathogen_population(starting_cluster):
    pathogens = []
    for _ in range(0, starting_cluster):
        pathogens.append(Generator(ngpu=environment.ngpu,
                    latent_vector_size=environment.latent_vector_size,
                    generator_latent_maps=64,
                    number_of_colors=environment.number_of_colors).to(environment.device))
    print('pathogens: ', [pathogen.random_tag for pathogen in pathogens])
    return pathogens


def cross_train_iteration(hosts, pathogens, host_type_selector):

    print('cross-training round with host type: %s' % host_type_selector)

    for host, pathogen in product(hosts[host_type_selector], pathogens):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena.cross_train(1)
        arena.sample_images()
        print(arena.generator_instance.random_tag, ': ', end='')
        print("real_err: %s, gen_err: %s" % arena.match())

    for host in hosts[host_type_selector]:
        print('host', host.random_tag, host.current_fitness, host.gen_error_map)

    for pathogen in pathogens:
        print('pathogen', pathogen.random_tag, pathogen.fitness_map)


def chain_progression(individuals_per_species, starting_cluster):
        # by default we will be starting with the weaker pathogens, at least for now
    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)
    cross_train_iteration(hosts, pathogens, 'light')
    cross_train_iteration(hosts, pathogens, 'PreLU')
    cross_train_iteration(hosts, pathogens, 'base')


def brute_force_training(restarts, epochs):
    print('bruteforcing restarts')
    hosts = spawn_host_population(restarts*3)['base']
    pathogens = spawn_pathogen_population(restarts)

    for host, pathogen in product(hosts, pathogens):
        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena.cross_train(epochs)
        arena.sample_images()
        print(arena.generator_instance.random_tag)
        print(arena.match())


if __name__ == "__main__":
    image_folder = "./image"
    image_size = 64
    number_of_colors = 1
    imtype = 'mnist'

    mnist_dataset = dset.MNIST(root=image_folder, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),]))

    environment = GANEnvironment(mnist_dataset)

    learning_rate = 0.0002
    beta1 = 0.5

    gen_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))
    disc_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))

    chain_progression(5, 5)
    brute_force_training(15, 5)


    # gen = Generator(ngpu=environment.ngpu,
    #                 latent_vector_size=environment.latent_vector_size,
    #                 generator_latent_maps=64,
    #                 number_of_colors=environment.number_of_colors).to(environment.device)
    #
    # disc = Discriminator(ngpu=environment.ngpu,
    #                      latent_vector_size=environment.latent_vector_size,
    #                      discriminator_latent_maps=64,
    #                      number_of_colors=environment.number_of_colors).to(environment.device)
    #
    #
    # arena = Arena(environment=environment,
    #               generator_instance=gen,
    #               discriminator_instance=disc,
    #               generator_optimizer_partial=gen_opt_part,
    #               discriminator_optimizer_partial=disc_opt_part,
    #               )
    #
    # arena.cross_train(5)
    # arena.sample_images()
    # print(arena.generator_instance.random_tag)
    # print(arena.match())
