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


trace_dump_file = 'run_trace.csv'
backflow_log = 'backflow.csv'

bruteforce_gen2disc = {}

def parse_past_runs(trace_dump_locations):

    master_table = []

    # debug_counter = -1
    with open(trace_dump_locations, 'r') as trace_dump_file:
        reader = csv.reader(trace_dump_file, delimiter='\t')
        for row in reader:
            # debug_counter -= 1
            # print(row)
            # if row[-1] == '2020-04-01T00:55:58.583021':
                # debug_counter = 5
            # if debug_counter == 0:
            #     raise Exception('debug')
            if row[0] == '>':  # enter master run
                master_table.append([row, ])
                # print('lvl1 enter')
                continue
            if row[0] == '>>':  # enter run sub-section
                master_table[-1].append([row, ])  # in the latest master run add a function
                # print('lvl2 enter')
                continue
            if row[0] == '>>>':  # enter run sub-sub-section
                master_table[-1][-1].append([row, ])  # and a sub-function
                # print('lvl3 enter')
                # print(master_table[-1][-1][-1])
                continue
            if row[0] == '<<<':  # exit run sub-sub-section
                master_table[-1][-1][-1].append(row)
                # print('lvl3 exit')
                # print(master_table[-1][-1][-1])
                continue
            if row[0] == '<<':  # exist sub-section
                master_table[-1][-1].append(row)
                # print('lvl2 exit')
                continue
            if row[0] == '<':  # exit master run
                master_table[-1].append(row)
                # print('lvl1 exit')
                continue
            # print('default add')
            master_table[-1][-1][-1].append(row)
            # print(master_table[-1][-1][-1])

    return master_table


def extract_bruteforce_data(bruteforce_run):
    duration = (datetime.fromisoformat(bruteforce_run[-1][2]) -
                datetime.fromisoformat(bruteforce_run[0][4])).total_seconds()/60.
    # duration = float(bruteforce_run[-1][-1])
    fid_collector = []
    tag_collector = []
    for data_dump_row in bruteforce_run[1]:
        if data_dump_row[0] == 'sampled images from':
            fid_collector.append(float(data_dump_row[-1]))
            tag_collector.append(data_dump_row[2])
        if data_dump_row[0] == 'post-cross-train and match':
            bruteforce_gen2disc[data_dump_row[4]] = data_dump_row[3]


    return fid_collector, tag_collector, duration


def extract_evo_data(chain_evolve_run):
    duration = (datetime.fromisoformat(chain_evolve_run[-1][2]) -
                datetime.fromisoformat(chain_evolve_run[0][4])).total_seconds()/60.
    # print('duration raw:', duration)
    # duration = float(chain_evolve_run[-1][-1])
    # print('duration corrected:', duration)
    final_pathogens_list = chain_evolve_run[-2][0][3]
    final_pathogens_list = final_pathogens_list[1:-1].split(', ')
    fid_collector = [-1]*len(final_pathogens_list)
    tag_collector = ["None"]*len(final_pathogens_list)
    for entry in chain_evolve_run[-2][1:-1]:
        if entry[0] == 'sampled images from':
            # print(entry)
            fid_collector[int(entry[1])] = float(entry[-1])
            tag_collector[int(entry[1])] = entry[2]

    return fid_collector, tag_collector, duration


def render_fid_performances(attribution_map):

    def draw_p_vals_table(_dataset):
        p_matrix = np.ones((len(_dataset), len(_dataset)))
        ratio_matrix = np.ones((len(_dataset), len(_dataset)))

        for (i_1, subset1), (i_2, subset2) in combinations(enumerate(_dataset), 2):
            _, pval = ttest_ind(subset1, subset2)
            p_matrix[i_1, i_2] = pval
            p_matrix[i_2, i_1] = pval

            ratio_matrix[i_1, i_2] = np.median(subset1)/np.median(subset2)
            ratio_matrix[i_2, i_1] = np.median(subset2)/np.median(subset1)

        # norm = mcol.BoundaryNorm([0., 0.01, 0.05, 0.1, 1.], ncolors=256)
        # plt.imshow(p_matrix, cmap='RdBu_r', interpolation=None, norm=norm)

        plt.imshow(ratio_matrix, cmap='RdBu', interpolation=None, vmin=0., vmax=2.)
        plt.colorbar()
        stat_sig = np.argwhere(p_matrix < 0.05).T
        plt.scatter(stat_sig[0], stat_sig[1], marker='*', c='k')


        locs, labels = plt.xticks()
        plt.xticks(locs, method_names)

        locs, labels = plt.yticks()
        plt.yticks(locs, [''] + method_names)

        plt.xticks(rotation=45)


    def draw_box_plot(_dataset):
        flatten = lambda l: [item for sublist in l for item in sublist]
        plt.boxplot(_dataset)
        x_pad = [[_i+1  # +random.random()/10.
                  for _ in range(len(_data))]
                 for _i, _data in enumerate(_dataset)]
        plt.scatter(flatten(x_pad), flatten(_dataset), c='k')
        locs, labels = plt.xticks()
        plt.xticks(locs, method_names)
        plt.xticks(rotation=45)

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

    plt.title('minimum fids achieved')
    draw_box_plot(best_fids_achieved)
    plt.show()

    plt.title('p values for minimum fids achieved')
    draw_p_vals_table(best_fids_achieved)
    plt.show()

    plt.title('standard fids achieved')
    draw_box_plot(all_fids_achieved)
    plt.show()

    plt.title('p values for standard fids achieved')
    draw_p_vals_table(all_fids_achieved)
    plt.show()

    plt.title('execution times')
    draw_box_plot(exec_times)
    plt.show()

    plt.title('fids vs execution time')
    # c_pad = [[i/len(exec_times)]*len(_data) for i, _data in enumerate(exec_times)]
    for i, lab in enumerate(method_names):
        plt.scatter(exec_times[i], best_fids_achieved[i], label=lab)
    plt.legend()
    plt.xlabel('execution time (mins)')
    plt.ylabel('minimal fid achieved')
    plt.show()


def pull_best_fid_tags(attribtution_map):
    method_names = []
    best_fid_gen_tags = []
    select_disc_tags = []

    for i, (key, value) in enumerate(attribution_map.items()):
        method_names.append(' '.join(key))
        best_fid_gen_tags.append([])
        for sub_key, (exec_time, fids, tags) in value.items():
            fids = np.array(fids)
            min_loc = fids.argmin(fids)
            best_fid_gen_tags[i].append(tags[min_loc])

    for method, load in zip(method_names, best_fid_gen_tags):
        if 'brute-force' in method:
            for best_brute_force_tag in load:
               select_disc_tags.append(bruteforce_gen2disc[best_brute_force_tag])

    return method_names, best_fid_gen_tags, select_disc_tags


def buffer_gen_disc_fid_tags(best_fid_gen_tags, select_disc_tags):
    with open(backflow_log, 'w') as target:
        writer = csv.writer(target, delimiter='\t')
        for gen_tag, disc_tag in product(best_fid_gen_tags, select_disc_tags):
            writer.writerow([gen_tag, disc_tag])


if __name__ == "__main__":

    run_skip = []

    master_table = parse_past_runs(trace_dump_file)
    attribution_map = defaultdict(dict)
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
                or sub_entry[0][1] == 'stochastic base round robin':
                extracted_fids, final_random_tags, duration = extract_evo_data(sub_entry)
            elif sub_entry[0][1] == 'matching from tags':  # TODO: extract the results of matches
                pass
            else:
                print(sub_entry[0])
                raise Exception('unknown selection structure: %s' % sub_entry[0][1])
            attribution_map[(sub_entry[0][1], sub_entry[0][2], sub_entry[0][3])][sub_entry[0][-1]] =\
                [duration, extracted_fids, final_random_tags]

            print('\t', i_1, i_2 + 1, sub_entry[0])
            # for i_3, sub_sub_entry in enumerate(sub_entry[1:-1]):
            #     print('\t\t', i_1, i_2 + 1, i_3 + 1, sub_sub_entry[0])
            #     print('\t\t', i_1, i_2 + 1, i_3 + 1, sub_sub_entry[-1])
            print('\t', i_1, i_2 + 1, sub_entry[-1])

        print(i_1, entry[-1])

    # pprint(dict(attribution_map))

    render_fid_performances(attribution_map)

    method_names, best_fid_gen_tags, select_disc_tags = pull_best_fid_tags(attribution_map)
    print('select disc tags:', select_disc_tags)
    print('best fid gen tags:', best_fid_gen_tags)
    buffer_gen_disc_fid_tags(best_fid_gen_tags, select_disc_tags)