import pickle
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
import csv

trace_dump_file = 'run_trace.csv'

def parse_past_runs(trace_dump_locations):

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
            if row[0] == '<<':  # exist sub-section
                master_table[-1][-1].append(row)
                continue
            if row[0] == '<':  # exit master run
                master_table[-1].append(row)
                continue
            master_table[-1][-1].append(row)

    return master_table


def extract_bruteforce_fids(bruteforce_run):
    fid_collector = []
    for data_dump_row in bruteforce_run:
        if data_dump_row[0] == 'sampled images from':
            fid_collector.append(float(data_dump_row[-1]))
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
    bruteforce = master_table[-1][1]
    brute_fids = extract_bruteforce_fids(bruteforce)
    pprint(brute_fids)