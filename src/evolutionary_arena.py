from src.gans.discriminators import Discriminator, Discriminator_PReLU, Discriminator_light
from src.gans.generators import Generator
from src.gans.train_and_match import Arena, GANEnvironment
import pickle
import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import combinations, product
import torch.optim as optim
import random
from collections import defaultdict
import os
from datetime import datetime, timedelta
import csv
from src.compute_fid import calc_single_fid


evo_trace_dump_location = "evolved_hosts_pathogen_map.dmp"
evo2_trace_dump_location = "evolved_2_hosts_pathogen_map.dmp"
brute_force_trace_dump_location = 'brute_force_pathogen_map.dmp'

trace_dump_file = 'run_trace.csv'


def dump_trace(payload_list):
    """
    Wrapper for writing execution trace to a master storage file for the analyisis/

    :param payload_list:
    :return:
    """
    if not os.path.isfile(trace_dump_file):
        open(trace_dump_file, 'w')

    with open(trace_dump_file, 'a') as destination:
        writer = csv.writer(destination, delimiter='\t')
        writer.writerow(payload_list)


def dump_with_backup(object, location):
    if os.path.isfile(location):
        new_location = location[:-4] + '-bckp-' + datetime.now().isoformat() + location[-4:]
        os.rename(location, new_location)
    pickle.dump(object, open(location, 'wb'))


def render_evolution(random_tags_list):
    """
    Helper functiton, to showo the tag change with training encounters for generators and
    discriminators

    :param random_tags_list: tag list for which to show the evolution
    :return:
    """
    for random_tag in random_tags_list:
        print(random_tag, ' <- ', end='')
    print()


class StopWatch(object):
    """
    Helper object to measure code execution time.
    """

    def __init__(self):
        self._t = timedelta(microseconds=0)
        self._start = datetime.now()

    def start(self):
        self._start = datetime.now()

    def stop(self):
        delta = datetime.now() - self._start
        self._t += delta
        self._start = datetime.now()

    def get_total_time(self):
        if self._t != 0:
            return self._t.total_seconds() / 60.
        else:
            return 0


def spawn_host_population(individuals_per_species):
    """
    Generates three populations of hosts/discriminators, based on the supplied population size


    :param individuals_per_species: population size
    :return: dict: host_type -> [hosts], with host_type in ['base', 'PreLu', 'light']
    """
    hosts = {
        'base': [],
        'PreLU': [],
        'light': []
    }
    for _ in range(0, individuals_per_species):
        hosts['base'].append(Discriminator(ngpu=environment.ngpu,
                         latent_vector_size=environment.latent_vector_size,
                         discriminator_latent_maps=64,
                         number_of_colors=environment.number_of_colors).to(environment.device))
        hosts['PreLU'].append(Discriminator_PReLU(ngpu=environment.ngpu,
                         latent_vector_size=environment.latent_vector_size,
                         discriminator_latent_maps=64,
                         number_of_colors=environment.number_of_colors).to(environment.device))
        hosts['light'].append(Discriminator_light(ngpu=environment.ngpu,
                                         latent_vector_size=environment.latent_vector_size,
                                         discriminator_latent_maps=32,
                                         number_of_colors=environment.number_of_colors).to(environment.device))

    for host_type, _hosts in hosts.items():
        print(host_type, ': ', [host.random_tag for host in _hosts])

    return hosts


def spawn_pathogen_population(starting_cluster):
    """
    Generate a population of pathogens/generators of size supplied by the parameter

    :param starting_cluster: pathogen population size
    :return: list of pathogens
    """
    pathogens = []
    for _ in range(0, starting_cluster):
        pathogens.append(Generator(ngpu=environment.ngpu,
                    latent_vector_size=environment.latent_vector_size,
                    generator_latent_maps=64,
                    number_of_colors=environment.number_of_colors).to(environment.device))

    print('pathogens: ', [pathogen.random_tag for pathogen in pathogens])
    return pathogens


def cross_train_iteration(hosts, pathogens, host_type_selector, epochs=1, timer=None):
    """
    Specialized, single iteration of round-robin in the population

    :param hosts: host population dict
    :param pathogens: list of pathogens
    :param host_type_selector: type of hosts to propagate in
    :param epochs: epochs - training epochs per infection / training round
    :param timer: optional StopWatch timer instance for run timing
    """

    dump_trace(['>>>', 'cross-train',
                [host.random_tag for host in hosts[host_type_selector]],
                [pathogen.random_tag for pathogen in pathogens],
                host_type_selector, epochs,
                datetime.now().isoformat()])

    print('cross-training round with host type: %s' % host_type_selector)

    for (host_no, host), (pathogen_no, pathogen) in product(enumerate(hosts[host_type_selector]),
                                                            enumerate(pathogens)):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        dump_trace(['pre-train:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag, ])

        arena.cross_train(epochs, timer=timer)

        arena.sample_images()

        current_fid = calc_single_fid(arena.generator_instance.random_tag)
        dump_trace(['sampled images from',
                    pathogen_no,
                    arena.generator_instance.random_tag, current_fid])

        arena_match_results = arena.match()


        dump_trace(['post-cross-train and match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.fitness_map.get(
                        arena.discriminator_instance.random_tag, 0.05)])

        print("%s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))

    for (host_no, host), (pathogen_no, pathogen) in product(enumerate(hosts[host_type_selector]),
                                                            enumerate(pathogens)):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena_match_results = arena.match(timer=timer)

        dump_trace(['final cross-match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.fitness_map.get(
                        arena.discriminator_instance.random_tag, 0.05)])

        print("%s vs %s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena.discriminator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))

    for host in hosts[host_type_selector]:
        print('host', host.random_tag, host.current_fitness, host.gen_error_map)

    for pathogen in pathogens:
        print('pathogen', pathogen.random_tag, pathogen.fitness_map)
        render_evolution(pathogen.tag_trace)

    dump_trace(['<<<', 'cross-train',
                datetime.now().isoformat()])


def round_robin_iteration(hosts, pathogens, host_type_selector, epochs=1,
                          rounds=None, randomized=False, timer=None):
    """
    Generalized round-robin round

    :param hosts: dict of hosts
    :param pathogens: list of pathogens
    :param host_type_selector: host type to propagate in
    :param epochs: epochs to train each encounter for
    :param rounds: total budget of encounters
    :param randomized: if round robin is randomized or deterministic
    :param timer: optional StopWatch instance to time the execution
    """

    dump_trace(['>>>', 'round-robin',
                [host.random_tag for host in hosts[host_type_selector]],
                [pathogen.random_tag for pathogen in pathogens],
                host_type_selector, epochs,
                datetime.now().isoformat()])

    if rounds is None:
        rounds = len(hosts) * len(pathogens)

    print('round-robin round with host type: %s' % host_type_selector)

    if randomized:
        hosts_no = random.choices(range(len(hosts[host_type_selector])), k=rounds)
        pathogens_no = random.choices(range(len(pathogens)), k=rounds)

        generator = zip(hosts_no, pathogens_no)

    else:
        generator_list = [(host_no, pathogen_no) for host_no, pathogen_no in
                          product(range(len(hosts[host_type_selector])), range(len(pathogens)))]
        repeats = int(rounds/len(generator_list))
        generator = generator_list*repeats

    for host_no, pathogen_no in generator:

        host = hosts[host_type_selector][host_no]
        pathogen = pathogens[pathogen_no]

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        dump_trace(['pre-train:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag, ])

        arena.cross_train(epochs, timer=timer)

        arena.sample_images()

        current_fid = calc_single_fid(arena.generator_instance.random_tag)
        dump_trace(['sampled images from',
                    pathogen_no,
                    arena.generator_instance.random_tag, current_fid])

        arena_match_results = arena.match(timer=timer)

        dump_trace(['post-cross-train and match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.fitness_map.get(
                        arena.discriminator_instance.random_tag, 0.05)])

        print("%s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))

    for (host_no, host), (pathogen_no, pathogen) in product(enumerate(hosts[host_type_selector]),
                                                            enumerate(pathogens)):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena_match_results = arena.match(timer=timer)

        dump_trace(['final cross-match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.fitness_map.get(
                        arena.discriminator_instance.random_tag, 0.05)])

        print("%s vs %s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena.discriminator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))

    for host in hosts[host_type_selector]:
        print('host', host.random_tag, host.current_fitness, host.gen_error_map)

    for pathogen in pathogens:
        print('pathogen', pathogen.random_tag, pathogen.fitness_map)
        render_evolution(pathogen.tag_trace)

    dump_trace(['<<<', 'round-robin',
                datetime.now().isoformat()])


def round_robin_deterministic(individuals_per_species, starting_cluster):
    """
    Deterministic round-robin wrapper

    :param individuals_per_species: host population size
    :param starting_cluster: pathogen population size
    """

    dump_trace(['>>', 'deterministic base round robin', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])

    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)

    timer = StopWatch()

    round_robin_iteration(hosts, pathogens, 'base',
                          1,
                          rounds=3*individuals_per_species*starting_cluster,
                          randomized=False, timer=timer)

    host_map = {}
    pathogen_map = {}
    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo_trace_dump_location)

    dump_trace(['<<', 'deterministic base round robin', datetime.now().isoformat(),
                timer.get_total_time()])


def round_robin_randomized(individuals_per_species, starting_cluster):
    """
    Randomized round-robin wrapper

    :param individuals_per_species: host population size
    :param starting_cluster: pathogen population size
    """

    dump_trace(['>>', 'stochastic base round robin', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])

    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)

    timer = StopWatch()

    cross_train_iteration(hosts, pathogens, 'base', 1, timer=timer)
    round_robin_iteration(hosts, pathogens, 'base',
                          1,
                          rounds=2*individuals_per_species*starting_cluster,
                          randomized=True,
                          timer=timer)

    host_map = {}
    pathogen_map = {}
    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo_trace_dump_location)

    dump_trace(['<<', 'stochastic base round robin', datetime.now().isoformat(),
                timer.get_total_time()])


def chain_progression(individuals_per_species, starting_cluster):
    """
    Deterministic round-robin with heterogenous naive populations transitions

    :param individuals_per_species: host population size
    :param starting_cluster: pathogen population size
    """

    dump_trace(['>>', 'chain progression', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])

    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)

    timer = StopWatch()

    cross_train_iteration(hosts, pathogens, 'light', 1, timer=timer)
    cross_train_iteration(hosts, pathogens, 'PreLU', 1, timer=timer)
    cross_train_iteration(hosts, pathogens, 'base', 3, timer=timer)

    host_map = {}
    pathogen_map = {}
    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo_trace_dump_location)

    dump_trace(['<<', 'chain progression', datetime.now().isoformat(),
                timer.get_total_time()])


def homogenous_chain_progression(individuals_per_species, starting_cluster):
    """
    Deterministic round-robin with homogenous naive populations transition

    :param individuals_per_species:
    :param starting_cluster:
    :return:
    """

    dump_trace(['>>', 'homogenous chain progression', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])

    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)

    timer = StopWatch()

    cross_train_iteration(hosts, pathogens, 'base', 1, timer=timer)
    cross_train_iteration(hosts, pathogens, 'base', 1, timer=timer)
    cross_train_iteration(hosts, pathogens, 'base', 3, timer=timer)

    host_map = {}
    pathogen_map = {}
    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo_trace_dump_location)

    dump_trace(['<<', 'homogenous chain progression', datetime.now().isoformat(),
                timer.get_total_time()])


def evolve_in_population(hosts_list, pathogens_list, pathogen_epochs_budget, fit_reset=False,
                         timer=None):
    """
    Evolution in population round

    :param hosts_list: list of hosts in the population
    :param pathogens_list: list of pathogens in the population
    :param pathogen_epochs_budget: budget for training by evolution, in epochs
    :param fit_reset: whether fitness is assumed to be reset upon the start of the evolution
    :param timer: optional StopWatch timer instance for execution duration timing
    """

    def pathogen_fitness_retriever(pathogen):
        fitness = 0.05
        try:
            fitness = max(pathogen.fitness_map.values())
        except ValueError:
            pass

        return fitness

    dump_trace(['>>>', 'evolve_in_population',
                [host.random_tag for host in hosts_list],
                [pathogen.random_tag for pathogen in pathogens_list],
                pathogen_epochs_budget,
                datetime.now().isoformat()])

    pathogens_index = list(range(0, len(pathogens_list)))
    hosts_index = list(range(0, len(hosts_list)))

    if fit_reset:
        pathogens_fitnesses = [20.]*len(pathogens_list)
        hosts_fitnesses = [1.]*len(hosts_list)
    else:
        pathogens_fitnesses = [pathogen_fitness_retriever(_pathogen) for _pathogen in pathogens_list]
        hosts_fitnesses = [_host.current_fitness for _host in hosts_list]

    host_idx_2_pathogens_carried = defaultdict(list)

    i = 0

    while i < pathogen_epochs_budget:
        if pathogen_epochs_budget < 0:
            dump_trace(['iterations budget overflow break'])
            break
        pathogen_epochs_budget -= 1

        print('current fitness tables: %s; %s' % (hosts_fitnesses, pathogens_fitnesses))
        dump_trace(['current tag/fitness tables',
                    [host.random_tag for host in hosts_list],
                    [pathogen.random_tag for pathogen in pathogens_list],
                    hosts_fitnesses, pathogens_fitnesses])

        current_host_idx = random.choices(hosts_index, weights=hosts_fitnesses)[0]
        current_pathogen_idx = random.choices(pathogens_index, weights=pathogens_fitnesses)[0]

        arena = Arena(environment=environment,
                  generator_instance=pathogens_list[current_pathogen_idx],
                  discriminator_instance=hosts_list[current_host_idx],
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena_match_results = arena.match(timer=timer)

        print("%s: real_err: %s, gen_err: %s; updated fitnesses: host: %s path: %s" % (
            arena.generator_instance.random_tag,
            arena_match_results[0], arena_match_results[1],
            arena.discriminator_instance.current_fitness,
            arena.generator_instance.fitness_map.get(arena.discriminator_instance.random_tag,
                                                     0.05)))

        dump_trace(['infection attempt:',
                    current_host_idx, current_pathogen_idx,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag, arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
            arena.generator_instance.fitness_map.get(arena.discriminator_instance.random_tag,
                                                     0.05)])

        if arena.generator_instance.fitness_map.get(arena.discriminator_instance.random_tag,
                                                    0.05) > 1:
            #infection
            if current_pathogen_idx not in host_idx_2_pathogens_carried[current_host_idx]:

                host_idx_2_pathogens_carried[current_host_idx].append(current_pathogen_idx)
                print('debug: host-pathogen mapping:', host_idx_2_pathogens_carried[
                    current_host_idx], current_host_idx, current_pathogen_idx)

            dump_trace(['infection successful, current host state:',
                        host_idx_2_pathogens_carried[current_host_idx],
                        arena.discriminator_instance.gen_error_map,
                        current_host_idx,
                        arena.discriminator_instance.current_fitness])

            if arena.discriminator_instance.current_fitness > 0.95 or \
                    arena.discriminator_instance.real_error > 0.1:
                #immune system is not bothered
                dump_trace(['silent infection'])
                arena.cross_train(gan_only=True, timer=timer)
                i += 0.5
            else:
                #immune sytem is active and competitive evolution happens:
                dump_trace(['full infection'])
                arena.cross_train(timer=timer)
                i += 1

            arena.sample_images()
            current_fid = calc_single_fid(arena.generator_instance.random_tag)

            dump_trace(['sampled images from', current_pathogen_idx,
                        arena.generator_instance.random_tag, current_fid])

            arena_match_results = arena.match(timer=timer)

            dump_trace(['post-infection',
                        current_host_idx, current_pathogen_idx,
                        arena.discriminator_instance.random_tag,
                        arena.generator_instance.random_tag,
                        arena_match_results[0], arena_match_results[1],
                        arena.discriminator_instance.current_fitness,
                        arena.generator_instance.fitness_map.get(
                            arena.discriminator_instance.random_tag, 0.05)])

            hosts_fitnesses[current_host_idx] = arena.discriminator_instance.current_fitness
            pathogens_fitnesses[current_pathogen_idx] = arena.generator_instance.fitness_map.get(
                arena.discriminator_instance.random_tag, 0.05)

        else:
            if current_pathogen_idx in host_idx_2_pathogens_carried[current_host_idx]:
                host_idx_2_pathogens_carried[current_host_idx].remove(current_pathogen_idx)

            hosts_fitnesses[current_host_idx] = arena.discriminator_instance.current_fitness

            try:
                pathogens_fitnesses[current_pathogen_idx] = max(
                    arena.generator_instance.fitness_map.values())
            except ValueError:
                pathogens_fitnesses[current_pathogen_idx] = 0.05

            dump_trace(['infection failed, current host state:',
                        host_idx_2_pathogens_carried[current_host_idx],
                        arena.discriminator_instance.gen_error_map,
                        current_host_idx,
                        arena.discriminator_instance.current_fitness])

    encountered_pathogens = []
    for (host_no, host), (pathogen_no, pathogen) in product(enumerate(hosts_list),
                                                            enumerate(pathogens_list)):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena_match_results = arena.match(timer=timer)

        if pathogen not in encountered_pathogens:

            arena.sample_images()
            current_fid = calc_single_fid(arena.generator_instance.random_tag)

            dump_trace(['sampled images from',
                        pathogen_no,
                        arena.generator_instance.random_tag,
                        current_fid])

            encountered_pathogens.append(pathogen)

        dump_trace(['final cross-match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.fitness_map.get(
                        arena.discriminator_instance.random_tag, 0.05)])

    dump_trace(['<<<', 'evolve_in_population', datetime.now().isoformat()])


def chain_evolve(individuals_per_species, starting_cluster):
    """
    Wrapper function for the evolution run

    :param individuals_per_species: host population size
    :param starting_cluster: pathogen population size
    """

    # by default we will be starting with the weaker pathogens, at least for now
    dump_trace(['>>', 'chain evolve', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])
    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)
    default_budget = individuals_per_species*starting_cluster

    timer = StopWatch()

    cross_train_iteration(hosts, pathogens, 'light', 1, timer=timer)
    evolve_in_population(hosts['light'], pathogens, default_budget, timer=timer)
    cross_train_iteration(hosts, pathogens, 'PreLU', 1, timer=timer)
    evolve_in_population(hosts['PreLU'], pathogens, default_budget, timer=timer)
    cross_train_iteration(hosts, pathogens, 'base', 1, timer=timer)
    evolve_in_population(hosts['base'], pathogens, default_budget, timer=timer)

    host_map = {}
    pathogen_map = {}

    #TODO: sample all the images, perform a massive cross-match

    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo2_trace_dump_location)
    # pickle.dump((host_map, pathogen_map), open('evolved_2_hosts_pathogen_map.dmp', 'wb'))
    dump_trace(['<<', 'chain evolve', datetime.now().isoformat(),
                timer.get_total_time()])


def chain_evolve_with_fitness_reset(individuals_per_species, starting_cluster):
    """
    Wrapper function for the evolution run with fitness reset upon evolution start

    :param individuals_per_species: host population size
    :param starting_cluster: pathogen population size
    """

    dump_trace(['>>', 'chain evolve fit reset', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])
    hosts = spawn_host_population(individuals_per_species)
    pathogens = spawn_pathogen_population(starting_cluster)
    default_budget = individuals_per_species*starting_cluster

    timer = StopWatch()

    cross_train_iteration(hosts, pathogens, 'light', 1, timer=timer)
    evolve_in_population(hosts['light'], pathogens, default_budget, fit_reset=True, timer=timer)
    cross_train_iteration(hosts, pathogens, 'PreLU', 1, timer=timer)
    evolve_in_population(hosts['PreLU'], pathogens, default_budget, fit_reset=True, timer=timer)
    cross_train_iteration(hosts, pathogens, 'base', 1, timer=timer)
    evolve_in_population(hosts['base'], pathogens, default_budget, fit_reset=True, timer=timer)

    host_map = {}
    pathogen_map = {}

    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo2_trace_dump_location)
    dump_trace(['<<', 'chain evolve fit reset', datetime.now().isoformat(),
                timer.get_total_time()])


def brute_force_training(restarts, epochs):
    """
    Reference training method

    :param restarts: number of restarts
    :param epochs:  number of epochs
    """
    dump_trace(['>>', 'brute-force',
                restarts, epochs,
                datetime.now().isoformat()])

    timer = StopWatch()

    dump_trace(['>>>', 'brute-force',
                restarts, epochs,
                datetime.now().isoformat()])

    print('bruteforcing starts')
    hosts = spawn_host_population(restarts)['base']
    pathogens = spawn_pathogen_population(restarts)

    for (host_no, host), (pathogen_no, pathogen) in zip(enumerate(hosts), enumerate(pathogens)):

        timer.start()

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        dump_trace(['pre-train:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag, ])

        arena.cross_train(epochs)

        timer.stop()

        arena.sample_images()

        current_fid = calc_single_fid(arena.generator_instance.random_tag)

        dump_trace(['sampled images from',
                    pathogen_no,
                    arena.generator_instance.random_tag,
                    current_fid])

        timer.start()

        arena_match_results = arena.match()

        timer.stop()

        dump_trace(['post-cross-train and match:',
                    host_no, pathogen_no,
                    arena.discriminator_instance.random_tag,
                    arena.generator_instance.random_tag,
                    arena_match_results[0], arena_match_results[1],
                    arena.discriminator_instance.current_fitness,
                    arena.generator_instance.fitness_map.get(
                        arena.discriminator_instance.random_tag, 0.05)])

        print("%s: real_err: %s, gen_err: %s" % (
            arena.generator_instance.random_tag,
            arena_match_results[0], arena_match_results[1]))

    for pathogen in pathogens:
        print(pathogen.random_tag, ": ", pathogen.fitness_map)

    pathogen_map = {}

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup(pathogen_map, brute_force_trace_dump_location)
    # pickle.dump(pathogen_map, open('brute_force_pathogen_map.dmp', 'wb'))
    dump_trace(['<<<', 'brute-force',
                datetime.now().isoformat()])
    dump_trace(['<<', 'brute-force',
                datetime.now().isoformat(),
                timer.get_total_time()])


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

    environment = GANEnvironment(mnist_dataset, device="cuda:1")

    learning_rate = 0.0002
    beta1 = 0.5

    gen_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))
    disc_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))

    dump_trace(['>', 'run started', datetime.now().isoformat()])

    round_robin_randomized(5, 5)
    round_robin_randomized(5, 5)
    round_robin_randomized(5, 5)
    round_robin_randomized(5, 5)
    round_robin_randomized(5, 5)

    round_robin_deterministic(5, 5)
    round_robin_deterministic(5, 5)
    round_robin_deterministic(5, 5)
    round_robin_deterministic(5, 5)
    round_robin_deterministic(5, 5)

    homogenous_chain_progression(5, 5)
    homogenous_chain_progression(5, 5)
    homogenous_chain_progression(5, 5)
    homogenous_chain_progression(5, 5)
    homogenous_chain_progression(5, 5)

    chain_progression(5, 5)
    chain_progression(5, 5)
    chain_progression(5, 5)
    chain_progression(5, 5)
    chain_progression(5, 5)

    chain_evolve_with_fitness_reset(3, 4)
    chain_evolve_with_fitness_reset(3, 4)
    chain_evolve_with_fitness_reset(3, 4)
    chain_evolve_with_fitness_reset(3, 4)
    chain_evolve_with_fitness_reset(3, 4)

    chain_evolve(3, 4)
    chain_evolve(3, 4)
    chain_evolve(3, 4)
    chain_evolve(3, 4)
    chain_evolve(3, 4)

    brute_force_training(10, 15)
    brute_force_training(10, 15)
    brute_force_training(10, 15)
    brute_force_training(10, 15)
    brute_force_training(10, 15)

    dump_trace(['<', 'run completed', datetime.now().isoformat()])