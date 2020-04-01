import pickle
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
import csv

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


def extract_bruteforce_fids(bruteforce_run):
    fid_collector = []
    for data_dump_row in bruteforce_run[1]:
        if data_dump_row[0] == 'sampled images from':
            fid_collector.append(float(data_dump_row[-1]))
    return fid_collector


def extract_final_evo_fids(chain_evolve_run):
    final_pathogens_list = chain_evolve_run[-2][0][3]
    final_pathogens_list = final_pathogens_list[1:-1].split(', ')
    fid_collector = [-1]*len(final_pathogens_list)
    for entry in chain_evolve_run[-2][1:-1]:
        if entry[0] == 'sampled images from':
            # print(entry)
            fid_collector[int(entry[1])] = float(entry[-1])
    return fid_collector


# fid_map, real_comparison = pickle.load(open('fid_scores.dmp', 'rb'))
#
# pprint(fid_map)
# pprint(real_comparison)
#
# host_map, pathogen_map = pickle.load(open('evolved_hosts_pathogen_map.dmp', 'rb'))
#
# evolution_trace = {}
#
# for pathogen_tag, (_, evo_list) in pathogen_map.items():
#     evolution_trace[pathogen_tag] = [fid_map.get(tag, -1) for tag in evo_list[1:]]
#
# pprint(host_map)
# pprint(real_comparison)
#
# pathogen_map2 = pickle.load(open('brute_force_pathogen_map.dmp', 'rb'))
#
# pprint(pathogen_map2)
#
# bruteforce_stats = []
# for pathogen in pathogen_map2.keys():
#     bruteforce_stats.append(fid_map[pathogen])
#
# evo_stats = []
# for pathogen in pathogen_map.keys():
#     evo_stats.append(fid_map[pathogen])
#
# # bruteforce_stats = np.array(bruteforce_stats)
# # evo_stats = np.array(evo_stats)
# plt.scatter([1]*len(bruteforce_stats), bruteforce_stats, c='k', marker='o', label='bruteforce')
# plt.scatter([2]*len(evo_stats),  evo_stats, c='r', marker='o', label='evolutionary')
# plt.legend()
#
# bruteforce_stats = np.array(bruteforce_stats)
# evo_stats = np.array(evo_stats)
# print('brutefroce: %.2f %.2f' % (np.mean(bruteforce_stats), np.std(bruteforce_stats)))
# print('evo: %.2f %.2f' % (np.mean(evo_stats), np.std(evo_stats)))
# print('t-test: %.2f p-val: %f' % ttest_ind(bruteforce_stats, evo_stats))
# plt.show()
#
# for key, trace in evolution_trace.items():
#     plt.plot(range(0, len(trace)), trace, label=key)
# plt.legend()
# plt.show()


if __name__ == "__main__":
    master_table = parse_past_runs(trace_dump_file)
    for i_1, entry in enumerate(master_table):
        print(i_1, entry[0])
        for i_2, sub_entry in enumerate(entry[1:-1]):
            print('\t', i_1, i_2 + 1, sub_entry[0])
            for i_3, sub_sub_entry in enumerate(sub_entry[1:-1]):
                print('\t\t', i_1, i_2 + 1, i_3 + 1, sub_sub_entry[0])
                print('\t\t', i_1, i_2 + 1, i_3 + 1, sub_sub_entry[-1])
            print('\t', i_1, i_2 + 1, sub_entry[-1])
        print(i_1, entry[-1])

    bruteforce = master_table[0][1]
    # print(bruteforce)
    brute_fids = extract_bruteforce_fids(bruteforce)
    pprint(brute_fids)

    bruteforce = master_table[2][2]
    # print(bruteforce)
    brute_fids = extract_bruteforce_fids(bruteforce)
    pprint(brute_fids)

    evo_1 = master_table[1][1]
    evo_1_fids = extract_final_evo_fids(evo_1)
    pprint(evo_1_fids)

    evo_2 = master_table[2][1]
    evo_2_fids = extract_final_evo_fids(evo_2)
    pprint(evo_2_fids)