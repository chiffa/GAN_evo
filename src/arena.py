from src.gans.discriminator_zoo import Discriminator, Discriminator_PReLU, Discriminator_light
from src.gans.generator_zoo import Generator
from src.gans.match_and_train import Arena, GANEnvironment
# from src.new_mongo_interface import save_pure_disc, save_pure_gen, filter_pure_disc, filter_pure_gen
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
from src.fid_analyser import calc_single_fid
from src.mongo_interface import save_pure_disc, save_pure_gen, update_pure_disc, update_pure_gen
from datetime import datetime
from configs import cuda_device
from configs import current_dataset as _dataset
from src.smtp_logger import logger, successfully_completed, smtp_error_bail_out
import smtplib


#Main Environment, where natural evolution algorithm is implemented


evo_trace_dump_location = "evolved_hosts_pathogen_map.dmp"
evo2_trace_dump_location = "evolved_2_hosts_pathogen_map.dmp"
brute_force_trace_dump_location = 'brute_force_pathogen_map.dmp'


def dump_trace(payload_list):
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
    for random_tag in random_tags_list:
        print(random_tag, ' <- ', end='')
    print()


def save_pure_disc_helper(discriminator_instance):
    save_pure_disc(discriminator_instance.save_instance_state())


def save_pure_gen_helpler(generator_instance):
    save_pure_gen(generator_instance.save_instance_state())


def update_pure_disc_helper(discriminator_instance):
    result = update_pure_disc(discriminator_instance.random_tag,
                 {'encounter_trace': discriminator_instance.encounter_trace,
                  'self_error': discriminator_instance.real_error,
                  'gen_error_map': discriminator_instance.gen_error_map,
                  'current_fitness': discriminator_instance.current_fitness})
    if result is None:
        save_pure_disc(discriminator_instance.save_instance_state())
        update_pure_disc(discriminator_instance.random_tag,
                 {'encounter_trace': discriminator_instance.encounter_trace,
                  'self_error': discriminator_instance.real_error,
                  'gen_error_map': discriminator_instance.gen_error_map,
                  'current_fitness': discriminator_instance.current_fitness})


def update_pure_gen_helper(generator_instance):
    result = update_pure_gen(generator_instance.random_tag,
                    {'encounter_trace': generator_instance.encounter_trace,
                     'fitness_map': generator_instance.fitness_map})
    if result is None:
        save_pure_gen(generator_instance.save_instance_state())
        update_pure_gen(generator_instance.random_tag,
                    {'encounter_trace': generator_instance.encounter_trace,
                     'fitness_map': generator_instance.fitness_map})

class StopWatch(object):

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


def cross_train_iteration(hosts, pathogens, host_type_selector, epochs=1, timer=None):

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

        arena.cross_train(epochs, timer=timer, commit=False)

        arena.sample_images()

        current_fid = calc_single_fid(arena.generator_instance.random_tag)
        dump_trace(['sampled images from',
                    pathogen_no,
                    arena.generator_instance.random_tag, current_fid])

        arena_match_results = arena.match(commit=False)


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
        
    
    #AMIR: why is the training implemented this way? why do we need a "final cross-match" train ?
    for (host_no, host), (pathogen_no, pathogen) in product(enumerate(hosts[host_type_selector]),
                                                            enumerate(pathogens)):

        arena = Arena(environment=environment,
                  generator_instance=pathogen,
                  discriminator_instance=host,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena_match_results = arena.match(timer=timer, commit=False)

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
        save_pure_disc_helper(host)

    for pathogen in pathogens:
        print('pathogen', pathogen.random_tag, pathogen.fitness_map)
        render_evolution(pathogen.tag_trace)
        save_pure_gen_helpler(pathogen)

    dump_trace(['<<<', 'cross-train',
                datetime.now().isoformat()])


def round_robin_iteration(hosts, pathogens, host_type_selector, epochs=1,
                          rounds=None, randomized=False, timer=None):

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
        # generator = product(range(len(hosts[host_type_selector])), range(len(pathogens)))
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

        arena.cross_train(epochs, timer=timer, commit=False)

        arena.sample_images()

        current_fid = calc_single_fid(arena.generator_instance.random_tag)
        dump_trace(['sampled images from',
                    pathogen_no,
                    arena.generator_instance.random_tag, current_fid])

        arena_match_results = arena.match(timer=timer, commit=False)

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

        arena_match_results = arena.match(timer=timer, commit=False)

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
        save_pure_disc_helper(host)

    for pathogen in pathogens:
        print('pathogen', pathogen.random_tag, pathogen.fitness_map)
        save_pure_gen_helpler(pathogen)
        render_evolution(pathogen.tag_trace)

    dump_trace(['<<<', 'round-robin',
                datetime.now().isoformat()])


def round_robin_deterministic(individuals_per_species, starting_cluster):

    dump_trace(['>>', 'deterministic base round robin 2', individuals_per_species, starting_cluster,
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
    # pickle.dump((host_map, pathogen_map), open('evolved_hosts_pathogen_map.dmp', 'wb'))

    dump_trace(['<<', 'deterministic base round robin 2', datetime.now().isoformat(),
                timer.get_total_time()])


def round_robin_randomized(individuals_per_species, starting_cluster):

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
    # pickle.dump((host_map, pathogen_map), open('evolved_hosts_pathogen_map.dmp', 'wb'))

    dump_trace(['<<', 'stochastic base round robin', datetime.now().isoformat(),
                timer.get_total_time()])


def chain_progression(individuals_per_species, starting_cluster):

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
    # pickle.dump((host_map, pathogen_map), open('evolved_hosts_pathogen_map.dmp', 'wb'))

    dump_trace(['<<', 'chain progression', datetime.now().isoformat(),
                timer.get_total_time()])


def homogenus_chain_progression(individuals_per_species, starting_cluster):

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
    # pickle.dump((host_map, pathogen_map), open('evolved_hosts_pathogen_map.dmp', 'wb'))

    dump_trace(['<<', 'homogenous chain progression', datetime.now().isoformat(),
                timer.get_total_time()])


def evolve_in_population(hosts_list, pathogens_list, pathogen_epochs_budget, fit_reset=False,
                         timer=None):

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

    iterations_limiter = pathogen_epochs_budget * 5

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

        # TODO: add a restart from a static state

        # TODO: realistically, we need to look in the carrying tables and then attempt to cross
        #  the infections to other, more efficient hosts.

        current_host_idx = random.choices(hosts_index, weights=hosts_fitnesses)[0]
        current_pathogen_idx = random.choices(pathogens_index, weights=pathogens_fitnesses)[0]

        arena = Arena(environment=environment,
                  generator_instance=pathogens_list[current_pathogen_idx],
                  discriminator_instance=hosts_list[current_host_idx],
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part)

        arena_match_results = arena.match(timer=timer, commit=False)

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
                # print('debug: pop evolve: silent infection')
                dump_trace(['silent infection'])
                arena.cross_train(gan_only=True, timer=timer, commit=False)
                i += 0.5
            else:
                #immune sytem is active and competitive evolution happens:
                # print('debug: pop evolve: full infection')
                dump_trace(['full infection'])
                arena.cross_train(timer=timer, commit=False)
                i += 1

            arena.sample_images()
            current_fid = calc_single_fid(arena.generator_instance.random_tag)

            dump_trace(['sampled images from', current_pathogen_idx,
                        arena.generator_instance.random_tag, current_fid])

            arena_match_results = arena.match(timer=timer, commit=False)

            dump_trace(['post-infection',
                        current_host_idx, current_pathogen_idx,
                        arena.discriminator_instance.random_tag,
                        arena.generator_instance.random_tag,
                        arena_match_results[0], arena_match_results[1],
                        arena.discriminator_instance.current_fitness,
                        arena.generator_instance.fitness_map.get(
                            arena.discriminator_instance.random_tag, 0.05)])

            # print("debug: pop evolve: post-train: real_err: %s, gen_err: %s" % (
                # arena_match_results[0], arena_match_results[1]))
            # print("debug: pop evolve: fitness map", arena.generator_instance.fitness_map)
            # print("debug: pop evolve: updated fitnesses: host: %s path: %s" % (
            #     arena.discriminator_instance.current_fitness,
            #     arena.generator_instance.fitness_map.get(
            #     arena.discriminator_instance.random_tag, 0.05)))

            hosts_fitnesses[current_host_idx] = arena.discriminator_instance.current_fitness
            pathogens_fitnesses[current_pathogen_idx] = arena.generator_instance.fitness_map.get(
                arena.discriminator_instance.random_tag, 0.05)

        else:
            # print('debug: pop evolve: infection fails')

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

        arena_match_results = arena.match(timer=timer, commit=False)

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

    for host in hosts_list:
        save_pure_disc_helper(host)

    for pathogen in pathogens_list:
        save_pure_gen_helpler(pathogen)

    dump_trace(['<<<', 'evolve_in_population', datetime.now().isoformat()])


def chain_evolve(individuals_per_species, starting_cluster):
    # by default we will be starting with the weaker pathogens, at least for now
    dump_trace(['>>', 'chain evolve', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])
    start = datetime.now()
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
    # by default we will be starting with the weaker pathogens, at least for now
    dump_trace(['>>', 'chain evolve fit reset', individuals_per_species, starting_cluster,
                datetime.now().isoformat()])
    start = datetime.now()
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

    #TODO: sample all the images, perform a massive cross-match

    for host in hosts['base']:
        host_map[host.random_tag] = [host.gen_error_map, host.current_fitness, host.real_error,
                                     host.tag_trace]

    for pathogen in pathogens:
        pathogen_map[pathogen.random_tag] = [pathogen.fitness_map, pathogen.tag_trace]

    dump_with_backup((host_map, pathogen_map), evo2_trace_dump_location)
    # pickle.dump((host_map, pathogen_map), open('evolved_2_hosts_pathogen_map.dmp', 'wb'))
    dump_trace(['<<', 'chain evolve fit reset', datetime.now().isoformat(),
                timer.get_total_time()])

def brute_force_training(restarts, epochs):
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


def match_from_tags(tag_pair_set):

    dump_trace(['>>', 'matching from tags',
                datetime.now().isoformat()])

    dump_trace(['>>>', 'matching from tags',
                datetime.now().isoformat()])

    tag_pair_set.reverse()

    disc_set = set()
    gen_set = set()
    for gen_tag, disc_tag in tag_pair_set:

        try:
            disc_set.add(disc_tag)
            gen_set.add(gen_tag)
            host = Discriminator(ngpu=environment.ngpu,
                             latent_vector_size=environment.latent_vector_size,
                             discriminator_latent_maps=64,
                             number_of_colors=environment.number_of_colors).to(environment.device)

            host.resurrect(disc_tag)

            pathogen = Generator(ngpu=environment.ngpu,
                                latent_vector_size=environment.latent_vector_size,
                                generator_latent_maps=64,
                                number_of_colors=environment.number_of_colors).to(environment.device)

            pathogen.resurrect(gen_tag)

            host.to(environment.device)
            pathogen.to(environment.device)

            arena = Arena(environment=environment,
                      generator_instance=pathogen,
                      discriminator_instance=host,
                      generator_optimizer_partial=gen_opt_part,
                      discriminator_optimizer_partial=disc_opt_part)

            arena_match_results = arena.match()

            dump_trace(['post-cross-train and match:',
                        disc_tag, gen_tag,
                        arena_match_results[0], arena_match_results[1]])

        except RuntimeError:
            pass

    dump_trace(['>>>', 'matching from tags',
                datetime.now().isoformat()])

    dump_trace(['<<', 'matching from tags',
                   datetime.now().isoformat()])


if __name__ == "__main__":
    image_folder = "./image"
    image_size = 64  # TODO: test how this parameter affects training stability
    number_of_colors = 1  # TODO: remember to adjust that
    if _dataset == 'cifar10':
        number_of_colors = 3
    imtype = _dataset

    date_time = datetime.now().strftime("%d.%m.%Y-%H.%M.%S")
    trace_dump_file = 'run_trace_' + date_time + '_' + imtype + '.csv'



    #TODO: here is where we are going for fashion-mnist; CIFAR-10 and CelebA datasets

    # # raw size: 218x178 @ 3c
    # celeb_a_dataset = dset.CelebA(root=image_folder, download=True,
    #                                 transform=transforms.Compose([
    #                                    transforms.Resize(image_size),
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.5,), (0.5,)),]))
    # raw size: 28x28 @ 1c
    fashion_mnist_dataset = dset.FashionMNIST(root=image_folder, download=True,
                                              transform=transforms.Compose([
                                           transforms.Resize(image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,)),]))

    # # raw size: 32x32 @ 3c
    cifar10_dataset = dset.CIFAR10(root=image_folder, download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),]))

    # raw size: 28x28 @ 1c
    mnist_dataset = dset.MNIST(root=image_folder, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),]))

    if _dataset == 'fashion_mnist':
        current_dataset = fashion_mnist_dataset
    elif _dataset == 'mnist':
        current_dataset = mnist_dataset
    elif _dataset == 'cifar10':
        current_dataset = cifar10_dataset
    else:
        raise Exception('unrecognized dataset type: %s' % _dataset)

    environment = GANEnvironment(current_dataset,
                                 device=cuda_device,
                                 number_of_colors=number_of_colors)

    learning_rate = 0.0002
    beta1 = 0.5

    gen_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))
    disc_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))

    dump_trace(['>', 'run started', datetime.now().isoformat()])

    run_start = datetime.now()

    try:
        # homogenus_chain_progression(5, 5)
        # homogenus_chain_progression(5, 5)
        # homogenus_chain_progression(5, 5)
        # homogenus_chain_progression(5, 5)
        # homogenus_chain_progression(5, 5)

        # chain_progression(5, 5)
        # chain_progression(5, 5)
        # chain_progression(5, 5)
        # chain_progression(5, 5)
        # chain_progression(5, 5)

        chain_evolve_with_fitness_reset(3, 3)
        chain_evolve_with_fitness_reset(3, 3)
        chain_evolve_with_fitness_reset(3, 3)
        chain_evolve_with_fitness_reset(3, 3)
        chain_evolve_with_fitness_reset(3, 3)

        # chain_evolve(3, 3)
        # chain_evolve(3, 3)
        # chain_evolve(3, 3)
        # chain_evolve(3, 3)
        # chain_evolve(3, 3)

        round_robin_randomized(5, 5)
        round_robin_randomized(5, 5)
        round_robin_randomized(5, 5)
        round_robin_randomized(5, 5)
        round_robin_randomized(5, 5)

        # round_robin_deterministic(5, 5)
        # round_robin_deterministic(5, 5)
        # round_robin_deterministic(5, 5)
        # round_robin_deterministic(5, 5)
        # round_robin_deterministic(5, 5)

        brute_force_training(10, 15)
        brute_force_training(10, 15)
        brute_force_training(10, 15)
        brute_force_training(10, 15)
        brute_force_training(10, 15)

        # brute_force_training(5, 30)
        # brute_force_training(5, 30)
        # brute_force_training(5, 30)
        # brute_force_training(5, 30)
        # brute_force_training(5, 30)

        # tag_pair_accumulator = []
        # with open('backflow.csv', 'r') as read_file:
        #     reader = csv.reader(read_file, delimiter='\t')
        #     for row in reader:
        #         if len(row) > 0:
        #             tag_pair_accumulator.append(row)
        #
        # match_from_tags(tag_pair_accumulator)

        # match_from_tags([('W0247GHVV4', '7TRW7286CW')])

        pass

    except Exception as exc:
        try:
            logger.error(exc, exc_info=True)
        except (AttributeError, smtplib.SMTPServerDisconnected):
            smtp_error_bail_out()
        raise

    dump_trace(['<', 'run completed', datetime.now().isoformat()])

    successfully_completed(start_date=run_start, start_device=cuda_device)

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
