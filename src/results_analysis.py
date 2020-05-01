import pickle
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcol
from scipy.stats import ttest_ind
import csv
from datetime import datetime
from collections import defaultdict
from itertools import combinations, product
import random
from matplotlib.lines import Line2D


trace_dump_file = 'run_trace.csv'
backflow_log = 'backflow.csv'

bruteforce_gen2disc = {}
bruteforce_disc2addchars = {}

# below the mapping convention is
# after_update_tag : [pre_update_tag, post_update_partner_tag]
gen_tag_trace = {}  # Maps the random tag to its previous state and disc that changed it
disc_tag_trace = {}  # Maps the random tag to its previous state and gen that changed it.

encounter_record = {}

master_fid_map = {}

disc_phases = {}


def parse_past_runs(trace_dump_locations):
    """
    Parses run records files into the master dictionary

    :param trace_dump_locations: location where the run records file is located
    :return: an embedded dictionary of the run records
    """

    master_table = []

    with open(trace_dump_locations, 'r') as trace_dump_file:
        reader = csv.reader(trace_dump_file, delimiter='\t')
        for row in reader:
            if row[0] == '>':  # enter master run
                master_table.append([row, ])
                continue
            if row[0] == '>>':  # enter run sub-section
                master_table[-1].append([row, ])  # in the latest master run add a function
                continue
            if row[0] == '>>>':  # enter run sub-sub-section
                master_table[-1][-1].append([row, ])  # and a sub-function=
                continue
            if row[0] == '<<<':  # exit run sub-sub-section
                master_table[-1][-1][-1].append(row)
                continue
            if row[0] == '<<':  # exist sub-section
                master_table[-1][-1].append(row)
                continue
            if row[0] == '<':  # exit master run
                master_table[-1].append(row)
                continue
            master_table[-1][-1][-1].append(row)

    return master_table


def extract_bruteforce_data(bruteforce_run):
    """
    Extracts the data from a reference run

    :param bruteforce_run: the embedded lists of run records for a reference run
    :return: final FIDs, final generators random tags, run duration
    """
    duration = (datetime.fromisoformat(bruteforce_run[-1][2]) -
                datetime.fromisoformat(bruteforce_run[0][4])).total_seconds()/60.
    try:
        duration = float(bruteforce_run[-1][-1])
    except:
        print('old format timing')
    fid_collector = []
    tag_collector = []
    for data_dump_row in bruteforce_run[1]:
        if data_dump_row[0] == 'sampled images from':
            fid_collector.append(float(data_dump_row[-1]))
            tag_collector.append(data_dump_row[2])
        if data_dump_row[0] == 'post-cross-train and match:':
            bruteforce_gen2disc[data_dump_row[4]] = data_dump_row[3]
            bruteforce_disc2addchars[data_dump_row[3]] = bruteforce_run[0][2:4]

    return fid_collector, tag_collector, duration


def extract_evo_data(chain_evolve_run):
    """
    Extracts the data from the evolutionary-structured run

    :param chain_evolve_run: the embedded lists of run records for an evolutionary-inspired run
    :return: final FIDs, final generators random tags, run duration
    """

    duration = (datetime.fromisoformat(chain_evolve_run[-1][2]) -
                datetime.fromisoformat(chain_evolve_run[0][4])).total_seconds()/60.
    try:
        duration = float(chain_evolve_run[-1][-1])
    except:
        print('old format timing')

    final_pathogens_list = chain_evolve_run[-2][0][3]
    final_pathogens_list = final_pathogens_list[1:-1].split(', ')
    fid_collector = [-1]*len(final_pathogens_list)
    tag_collector = ["None"]*len(final_pathogens_list)

    pre_train_buffer = []

    for entry in chain_evolve_run[-2][1:-1]:  # here we are pulling hte data from the last run alone

        if entry[0] == 'sampled images from':
            fid_collector[int(entry[1])] = float(entry[-1])
            tag_collector[int(entry[1])] = entry[2]

    for sub_run in chain_evolve_run[1:-1]:
        for entry in sub_run[1:-1]:

            if entry[0] == 'sampled images from':
                master_fid_map[entry[2]] = float(entry[-1])

            if entry[0] in ['infection attempt:', 'pre-train:']:
                pre_train_buffer = entry

            if entry[0] in ['post-cross-train and match:', 'post-infection']:

                gen_tag_trace[entry[4]] = [pre_train_buffer[4], entry[3]]
                disc_tag_trace[entry[3]] = [pre_train_buffer[3], entry[4]]
                encounter_record[(entry[3], entry[4])] = (float(entry[5]), float(entry[6]))

                disc_phases[entry[3]] = [sub_run[0][1], None]
                if sub_run[0][1] == 'cross-train':
                    disc_phases[entry[3]][1] = sub_run[0][4]

    return fid_collector, tag_collector, duration


def render_fid_performances(attribution_map):
    """
    Draws figures specifying performance per FID (boxplots and heatmaps)

    :param attribtution_map: master method+parameters -> runs reccords
    """

    def draw_p_vals_table(_dataset):
        p_matrix = np.ones((len(_dataset), len(_dataset)))
        ratio_matrix = np.ones((len(_dataset), len(_dataset)))

        for (i_1, subset1), (i_2, subset2) in combinations(enumerate(_dataset), 2):
            _, pval = ttest_ind(subset1, subset2)
            p_matrix[i_1, i_2] = pval
            p_matrix[i_2, i_1] = pval

            ratio_matrix[i_1, i_2] = np.median(subset1) / np.median(subset2)
            ratio_matrix[i_2, i_1] = np.median(subset2) / np.median(subset1)

        # norm = mcol.BoundaryNorm([0., 0.01, 0.05, 0.1, 1.], ncolors=256)
        # plt.imshow(p_matrix, cmap='RdBu_r', interpolation=None, norm=norm)

        plt.imshow(ratio_matrix, cmap='RdYlGn', interpolation=None, vmin=0., vmax=2.)
        plt.colorbar()
        stat_sig = np.argwhere(p_matrix < 0.05).T
        plt.scatter(stat_sig[0], stat_sig[1], marker='*', c='k')

        _method_names = [name_remap[method] for method in method_names]

        plt.xticks(range(len(_method_names)), _method_names)
        plt.yticks(range(len(_method_names)), _method_names)

        plt.xticks(rotation=45, rotation_mode="anchor", ha="right")


    def draw_box_plot(_dataset):
        flatten = lambda l: [item for sublist in l for item in sublist]

        for name, data in zip(method_names, _dataset):
            load = [name_remap[name], '&',
                    '%.2f' % np.median(data), '&',
                    '%.2f' % np.mean(data), '&',
                    '%.2f' % np.std(data), '\\']
            print(' '.join(load))

        plt.boxplot(_dataset)
        x_pad = [[_i+1  # +random.random()/10.
                  for _ in range(len(_data))]
                 for _i, _data in enumerate(_dataset)]
        plt.scatter(flatten(x_pad), flatten(_dataset), c='k')
        locs, labels = plt.xticks()
        _method_names = [name_remap[method] for method in method_names]
        plt.xticks(locs, _method_names)
        plt.xticks(rotation=45, rotation_mode="anchor", ha="right")


    method_names = []
    best_fids_achieved = []
    all_fids_achieved = []
    exec_times = []

    for i, (key, value) in enumerate(attribution_map.items()):
        method_names.append(' '.join(key))
        best_fids_achieved.append([])
        exec_times.append([])
        all_fids_achieved.append([])
        for sub_key, (exec_time, fids, tags) in value.items():
            best_fids_achieved[i].append(min(fids))
            exec_times[i].append(exec_time)
            all_fids_achieved[i] += fids

    plt.title('Minimum FID achieved per run by a method')
    draw_box_plot(best_fids_achieved)
    plt.show()

    plt.title('Relative performance of methods for best FID achieved')
    draw_p_vals_table(best_fids_achieved)
    plt.show()

    plt.title('FID of all generators per method')
    draw_box_plot(all_fids_achieved)
    plt.show()

    plt.title('Relative performance of methods overall')
    draw_p_vals_table(all_fids_achieved)
    plt.show()

    plt.title('Single run time for each method')
    draw_box_plot(exec_times)
    plt.show()

    plt.title('Best FID per run vs run time')
    # c_pad = [[i/len(exec_times)]*len(_data) for i, _data in enumerate(exec_times)]
    for i, lab in enumerate(method_names):
        plt.scatter(exec_times[i], best_fids_achieved[i], label=name_remap[lab])
    plt.legend()
    plt.xlabel('run time (mins)')
    plt.ylabel('minimal FID achieved')
    plt.show()


def pull_best_fid_tags(attribtution_map):
    """
    Extracts the best performing fids and tags associatated to them for each method from the
    attribution map

    :param attribtution_map: master method+parameters -> runs reccords
    :return: method names list, random tags of generators that achieved best fids, disc from
    reference runs that trained the best-performing generators in each run.
    """
    method_names = []
    best_fid_gen_tags = []
    select_disc_tags = []

    for i, (key, value) in enumerate(attribution_map.items()):
        method_names.append(' '.join(key))
        best_fid_gen_tags.append([])
        for sub_key, (exec_time, fids, tags) in value.items():
            fids = np.array(fids)
            min_loc = np.argmin(fids)
            best_fid_gen_tags[i].append(tags[min_loc])

    # pprint(bruteforce_gen2disc)

    for method, load in zip(method_names, best_fid_gen_tags):
        if 'brute-force' in method:
            for best_brute_force_tag in load:
               select_disc_tags.append(bruteforce_gen2disc[best_brute_force_tag])

    return method_names, best_fid_gen_tags, select_disc_tags


def render_training_history(method_names, best_fid_gen_tags):
    """
    Renders training history for all the methods and best FID generator tags

    :param method_names: names and specifications of runs for methods used to train the the FIDs
    :param best_fid_gen_tags:
    """

    type_map = {0: 'X',
                1: '*',
                2: 'P',
                3: 'X'}

    colors_map = {1: 'k',
                  2: 'b'}

    def render_fid_progression(method, gen_tags_trace):

        plt.title(name_remap[method])

        cmap = plt.get_cmap('RdYlGn')
        # print('gtt', gen_tags_trace)
        # print('gtt[0]', gen_tags_trace[0])
        # print('gtt[0][0]', gen_tags_trace[0][0])
        fids_per_line = [gen_line[0][-1] for gen_line in gen_tags_trace]
        # print('fpl', fids_per_line)

        fids_per_line = np.array(fids_per_line)

        lines_colors = (fids_per_line - np.min(fids_per_line)) /\
                       (np.max(fids_per_line) - np.min(fids_per_line))

        lines_colors = np.array(range(len(fids_per_line))) / float(len(fids_per_line))

        argsort = np.argsort(-fids_per_line)

        lines_colors = [cmap(fid) for fid in lines_colors[argsort]]

        type_idx_map = {}
        c = 0

        c_t_annotation_map = {}
        shown_c = []
        shown_t = []

        secondary_label_buffer = []
        secondary_label_buffer_2 = []

        for l_color, gen_line in zip(lines_colors, gen_tags_trace):
            root = gen_line[0][0]
            fids = [fid for _, _, _, _, fid in reversed(gen_line)]
            xs = range(len(fids))
            plt.plot(xs, fids,
                     # label=root,
                     c=l_color,
                     # linewidth=3
                     )

            disc_state = [disc_phases[disc_tag] for _, disc_tag, _, _, _ in gen_line]
            # print(disc_state)
            color_list = []
            type_list = []
            t = 0
            c_mem = ''
            cross_train_counter = 0
            print('new train')

            for state in reversed(disc_state):
                print('\ts', state)
                if state[0] == 'cross-train':
                    cross_train_counter += 1
                if state[1] is not None and c_mem != state[1] \
                        or cross_train_counter % 6 == 0:  # temporary
                    cross_train_counter = 1
                    print('xt', c_mem, '->', state[1], ':', t)
                    t += 1
                    c_mem = state[1]
                    c_t_annotation_map[type_map[t]] = c_mem
                print('\tt', c_mem, ':', t)
                type_list.append(type_map[t])

                if state[0] not in type_idx_map.keys():
                    print('xc', state[0], '! in', type_idx_map.keys())
                    c += 1
                    type_idx_map[state[0]] = c
                    c_t_annotation_map[colors_map[type_idx_map[state[0]]]] = state[0]
                print('\tc', state[0], ':', type_idx_map[state[0]])
                color_list.append(colors_map[type_idx_map[state[0]]])

            print(xs, fids)

            for _x, _f, _t, _c in zip(xs, fids, type_list, color_list):

                if _t not in shown_t or _c not in shown_c:
                    shown_t.append(_t)
                    shown_c.append(_c)

                    plt.plot(_x, _f, marker=_t, c=_c,
                             markersize=8)

                else:
                    plt.plot(_x, _f, marker=_t, c=_c, markersize=8)


        pprint(c_t_annotation_map)

        if len(c_t_annotation_map.keys()) == 0:
            print('problem detected')
            pass
        plt.ylabel('FID')
        plt.xlabel('encounter')

        legend_elements = []
        all_types = []
        all_colors = []

        for elt in type_map.values():
            if elt in c_t_annotation_map.keys() and elt not in all_types:
                all_types.append(elt)
                legend_elements.append(Line2D([0], [0],
                                  marker=elt,
                                  color='w',
                                  label=c_t_annotation_map[elt],
                                  markerfacecolor='k',
                                  markersize=10))

        if len(all_types) == 0:
            legend_elements.append(Line2D([0], [0],
                                  marker='x',
                                  color='w',
                                  label='base',
                                  markerfacecolor='k',
                                  markersize=10))

        for elt in colors_map.values():
            if elt in c_t_annotation_map.keys():
                print('color legend elt:', elt)
                legend_elements.append(Line2D([0], [0],
                                  marker='s',
                                  color='w',
                                  label=c_t_annotation_map[elt],
                                  markerfacecolor=elt,
                                  markersize=10))

        plt.legend(handles=legend_elements)
        plt.show()

    method_names = []
    run_tags = []

    for i, (key, value) in enumerate(attribution_map.items()):
        if key[0] == 'brute-force':
            continue
        method_names.append(' '.join(key))
        run_tags.append([])
        for sub_key, (exec_time, fids, tags) in value.items():
            run_tags[-1].append(tags)

    master_tag_trace = []

    # pprint(run_tags)

    for method, tag_sets in zip(method_names, run_tags):
        local_tag_trace = []

        # print(method)

        # print(tag_sets)

        for tag_set in tag_sets:  # we have the last tags performances
            tag_set_trace = []
            tag_set_disc_trace = []

            # print(tag_set)

            for tag in tag_set:
                single_tag_trace = []
                temp_tag = tag

                while temp_tag in gen_tag_trace.keys():  # does not enter if the tag is wrong
                    # print(temp_tag)
                    current_disc = gen_tag_trace[temp_tag][1]
                    single_tag_trace.append([temp_tag,
                                             current_disc,
                                             *encounter_record[current_disc, temp_tag],
                                             master_fid_map[temp_tag]])
                    temp_tag = gen_tag_trace[temp_tag][0]
                tag_set_trace.append(single_tag_trace)

                single_disc_trace = []
                # print(temp_tag, tag)
                disc_trace_root = gen_tag_trace[tag][1]  # we have a problem where the
                # evolutionary elts that have not evolved in the last round are not used.
                temp_disc = disc_trace_root

                while temp_disc in disc_tag_trace.keys():
                    single_disc_trace.append(temp_disc)
                    temp_disc = disc_tag_trace[temp_disc][0]

                tag_set_disc_trace.append(single_disc_trace)

            local_tag_trace.append([tag_set_trace, tag_set_disc_trace])
        master_tag_trace.append(local_tag_trace)


    for method, method_specific_tag_trace in zip(method_names, master_tag_trace):
        for gen_tags_trace, disc_tags_trace in method_specific_tag_trace:
            disc_idx = {}

            for i, disc_line in enumerate(disc_tags_trace):
                disc_idx.update(dict((tag, i) for tag in disc_line))

            for gen_tag_line in gen_tags_trace:
                print(gen_tag_line[0][0])
                for entry in reversed(gen_tag_line):
                    # print(entry)
                    print('\t%s - %s \t %.2e \t %.2e \t %.2f' % tuple(entry))

            render_fid_progression(method, gen_tags_trace)


if __name__ == "__main__":

    run_skip = []

    master_table = parse_past_runs(trace_dump_file)
    attribution_map = defaultdict(dict)

    collector_list = []

    for i_1, entry in enumerate(master_table):
        print(i_1, entry[0])
        if i_1 in run_skip:
            print('skipping')
            continue
        print('not skipping')
        for i_2, sub_entry in enumerate(entry[1:-1]):
            print(sub_entry[0])
            if sub_entry[0][1] == 'brute-force':
                extracted_fids, final_random_tags, duration = extract_bruteforce_data(sub_entry)
            elif sub_entry[0][1] == 'chain evolve' \
                or sub_entry[0][1] == 'chain progression' \
                or sub_entry[0][1] == 'chain evolve fit reset' \
                or sub_entry[0][1] == 'deterministic base round robin' \
                or sub_entry[0][1] == 'stochastic base round robin' \
                or sub_entry[0][1] == 'homogenous chain progression':
                extracted_fids, final_random_tags, duration = extract_evo_data(sub_entry)
                continue
            else:
                print(sub_entry[0])
                raise Exception('unknown selection structure: %s' % sub_entry[0][1])
            attribution_map[(sub_entry[0][1], sub_entry[0][2], sub_entry[0][3])][sub_entry[0][-1]] =\
                [duration, extracted_fids, final_random_tags]

            print('\t', i_1, i_2 + 1, sub_entry[0])
            print('\t', i_1, i_2 + 1, sub_entry[-1])

        print(i_1, entry[-1])


    print(attribution_map.keys())

    attribution_map_filter = []  # any keys for the attribution_map added to the list will be deleted

    name_remap = {
        ('stochastic base round robin 5 5'): 'stochastic round robin',
        ('chain evolve 3 4'): 'evolution with\nheterogeneous\npopulation jumps',
        ('brute-force 10 15'): 'reference',
        ('chain progression 5 5'): 'round-robin with\nheterogeneous\npopulation jumps',
        ('deterministic base round robin 5 5'): 'standard round robin',
        ('homogenous chain progression 5 5'): 'round-robin with\npopulation jumps'
    }

    for key in attribution_map_filter:
        del attribution_map[key]

    render_fid_performances(attribution_map)

    method_names, best_fid_gen_tags, select_disc_tags = pull_best_fid_tags(attribution_map)

    render_training_history(method_names, best_fid_gen_tags)
