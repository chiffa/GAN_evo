from src.gans.discriminator_zoo import Discriminator, Discriminator_PReLU, Discriminator_light
from src.gans.generator_zoo import Generator
from src.gans.trainer_zoo import Arena, GANEnvironment
# from src.new_mongo_interface import save_pure_disc, save_pure_gen, filter_pure_disc, filter_pure_gen
import pickle
import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import combinations, product
import torch.optim as optim
import random
from collections import defaultdict


def render_evolution(random_tags_list):
    for random_tag in random_tags_list:
        print(random_tag, ' <- ', end='')
    print()


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


def cross_train_iteration(hosts, pathogens, host_type_selector, epochs=1):

    print('cross-training round with host type: %s' % host_type_selector)

    for host, pathogen in product(hosts[host_type_selector], pathogens):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena.cross_train(epochs)
        arena.sample_images()
        arena_match_results = arena.match()
        print("%s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))

    for host, pathogen in product(hosts[host_type_selector], pathogens):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena_match_results = arena.match()

        print("%s vs %s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena.discriminator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))

    for host in hosts[host_type_selector]:
        print('host', host.random_tag, host.current_fitness, host.gen_error_map)

    for pathogen in pathogens:
        print('pathogen', pathogen.random_tag, pathogen.fitness_map)
        render_evolution(pathogen.tag_trace)

def chain_progression(individuals_per_species, starting_cluster):
        # by default we will be starting with the weaker pathogens, at least for now
    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)
    cross_train_iteration(hosts, pathogens, 'light', 1)
    cross_train_iteration(hosts, pathogens, 'PreLU', 1)
    cross_train_iteration(hosts, pathogens, 'base', 3)

    host_map = {}
    pathogen_map = {}
    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    pickle.dump((host_map, pathogen_map), open('evolved_hosts_pathogen_map.dmp', 'wb'))


def evolve_in_population(hosts_list, pathogens_list, pathogen_epochs_budget):
    #TODO: check infection decision correctness
    #TODO: add FID only on the most recent filter

    pathogens_index = list(range(0, len(pathogens_list)))
    pathogens_fitnesses = [1.]*len(pathogens_list)

    hosts_index = list(range(0, len(hosts_list)))
    hosts_fitnesses = [1.]*len(hosts_list)

    host_idx_2_pathogens_carried = defaultdict(list)

    i = 0

    while i < pathogen_epochs_budget:
        print('current fitness tables: %s; %s' % (hosts_fitnesses, pathogens_fitnesses))
        current_host_idx = random.choices(hosts_index, weights=hosts_fitnesses)[0]
        current_pathogen_idx = random.choices(pathogens_index, weights=pathogens_fitnesses)[0]

        arena = Arena(environment=environment,
                  generator_instance=pathogens_list[current_pathogen_idx],
                  discriminator_instance=hosts_list[current_host_idx],
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena_match_results = arena.match()
        print("%s: real_err: %s, gen_err: %s; updated fitnesses: host: %s path: %s" % (
            arena.generator_instance.random_tag,
            arena_match_results[0], arena_match_results[1],
            arena.discriminator_instance.current_fitness,
            arena.generator_instance.fitness_map.get(arena.discriminator_instance.random_tag, -1)))


        if arena.generator_instance.fitness_map.get(arena.discriminator_instance.random_tag, -1) > 1:
            #infection
            if current_pathogen_idx not in host_idx_2_pathogens_carried[current_host_idx]:
                host_idx_2_pathogens_carried[current_host_idx].append(current_pathogen_idx)
            if arena.discriminator_instance.current_fitness > 0.95:
                #immune system is not bothered
                print('debug: pop evolve: silent infection')
                arena.cross_train(gan_only=True)
                i += 0.5
            else:
                #immune sytem is active and competitive evolution happens:
                print('debug: pop evolve: full infection')
                arena.cross_train()
                i += 1

            arena.sample_images()
            arena_match_results = arena.match()

            print("debug: pop evolve: post-train: real_err: %s, gen_err: %s" % (
                arena_match_results[0], arena_match_results[1]))
            print("debug: pop evolve: fitness map", arena.generator_instance.fitness_map)
            print("debug: pop evolve: updated fitnesses: host: %s path: %s" % (
                arena.discriminator_instance.current_fitness,
                arena.generator_instance.fitness_map.get(arena.discriminator_instance.random_tag, -1)))

            hosts_fitnesses[current_host_idx] = max(
                0.01, arena.discriminator_instance.current_fitness)
            pathogens_fitnesses[current_pathogen_idx] = max(0.01,
            arena.generator_instance.fitness_map.get(
                arena.discriminator_instance.random_tag, -1))

        else:
            print('debug: pop evolve: infection fails')

            if current_pathogen_idx in host_idx_2_pathogens_carried[current_host_idx]:
                host_idx_2_pathogens_carried[current_host_idx].remove(current_pathogen_idx)

            hosts_fitnesses[current_host_idx] = max(
                0.01, arena.discriminator_instance.current_fitness)


def chain_evolve(individuals_per_species, starting_cluster):
    # by default we will be starting with the weaker pathogens, at least for now
    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)
    default_budget = individuals_per_species*starting_cluster
    cross_train_iteration(hosts, pathogens, 'light', 1)
    evolve_in_population(hosts['light'], pathogens, default_budget)
    cross_train_iteration(hosts, pathogens, 'PreLU', 1)
    evolve_in_population(hosts['PreLU'], pathogens, default_budget)
    cross_train_iteration(hosts, pathogens, 'base', 1)
    evolve_in_population(hosts['base'], pathogens, default_budget)

    host_map = {}
    pathogen_map = {}
    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    pickle.dump((host_map, pathogen_map), open('evolved_2_hosts_pathogen_map.dmp', 'wb'))

def brute_force_training(restarts, epochs):
    print('bruteforcing starts')
    hosts = spawn_host_population(restarts)['base']
    pathogens = spawn_pathogen_population(restarts)


    for host, pathogen in zip(hosts, pathogens):
        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena.cross_train(epochs)
        arena.sample_images()
        arena_match_results = arena.match()
        print("%s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))

    for pathogen in pathogens:
        print(pathogen.random_tag, ": ", pathogen.fitness_map)

    pathogen_map = {}

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    pickle.dump(pathogen_map, open('brute_force_pathogen_map.dmp', 'wb'))

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

    environment = GANEnvironment(mnist_dataset, device="cuda:0")

    learning_rate = 0.0002
    beta1 = 0.5

    gen_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))
    disc_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))

    # chain_progression(5, 5)
    chain_evolve(5, 5)
    # brute_force_training(5, 15)


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
