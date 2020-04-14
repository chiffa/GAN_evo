import pickle
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
import csv
from datetime import datetime
from collections import defaultdict


trace_dump_file = 'run_trace.csv'

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
    fid_collector = []
    tag_collector = []
    for data_dump_row in bruteforce_run[1]:
        if data_dump_row[0] == 'sampled images from':
            fid_collector.append(float(data_dump_row[-1]))
            tag_collector.append(data_dump_row[2])
    return fid_collector, tag_collector, duration


def extract_evo_data(chain_evolve_run):
    duration = (datetime.fromisoformat(chain_evolve_run[-1][2]) -
                datetime.fromisoformat(chain_evolve_run[0][4])).total_seconds()/60.
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
    title_pad = []
    data_pad = []
    full_data_pad = []
    exec_times = []

    for i, (key, value) in enumerate(attribution_map.items()):
        title_pad.append([' '.join(key)])
        data_pad.append([])
        exec_times.append([])
        full_data_pad.append([])
        for sub_key, (exec_time, fids, tags) in value.items():
            data_pad[i].append(min(fids))
            exec_times[i].append(exec_time)
            full_data_pad[i] += fids

    flatten = lambda l: [item for sublist in l for item in sublist]

    plt.title('minimum fids achieved')
    plt.boxplot(data_pad)
    x_pad = [[i+1]*len(_data) for i, _data in enumerate(data_pad)]
    plt.scatter(flatten(x_pad), flatten(data_pad), c='k')
    locs, labels = plt.xticks()
    plt.xticks(locs, title_pad)
    plt.xticks(rotation=45)

    plt.show()

    plt.title('standard fids achieved')
    plt.boxplot(full_data_pad)
    x_pad = [[i+1]*len(_data) for i, _data in enumerate(full_data_pad)]
    plt.scatter(flatten(x_pad), flatten(full_data_pad), c='k')
    locs, labels = plt.xticks()
    plt.xticks(locs, title_pad)
    plt.xticks(rotation=45)

    plt.show()

    plt.title('execution_times')
    plt.boxplot(exec_times)
    x_pad = [[i+1]*len(_data) for i, _data in enumerate(exec_times)]
    plt.scatter(flatten(x_pad), flatten(exec_times), c='k')
    locs, labels = plt.xticks()
    plt.xticks(locs, title_pad)
    plt.xticks(rotation=45)

    plt.show()


if __name__ == "__main__":
    master_table = parse_past_runs(trace_dump_file)
    attribution_map = defaultdict(dict)
    for i_1, entry in enumerate(master_table):
        print(i_1, entry[0])
        for i_2, sub_entry in enumerate(entry[1:-1]):
            print(sub_entry[0])
            if sub_entry[0][1] == 'brute-force':
                extracted_fids, final_random_tags, duration = extract_bruteforce_data(sub_entry)
            elif sub_entry[0][1] == 'chain evolve' or sub_entry[0][1] == 'chain progression':
                extracted_fids, final_random_tags, duration = extract_evo_data(sub_entry)
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
    # TODO: export the most interesting matches from different sources to a separate folder,
    #  then pull from new_arena the tags and make them compete there.

