import pickle
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

fid_map, real_comparison = pickle.load(open('fid_scores.dmp', 'rb'))

pprint(fid_map)
pprint(real_comparison)

host_map, pathogen_map = pickle.load(open('evolved_hosts_pathogen_map.dmp', 'rb'))

pprint(host_map)
pprint(real_comparison)

pathogen_map2 = pickle.load(open('brute_force_pathogen_map.dmp', 'rb'))

pprint(pathogen_map2)

bruteforce_stats = []
for pathogen in pathogen_map2.keys():
    bruteforce_stats.append(fid_map[pathogen])

evo_stats = []
for pathogen in pathogen_map.keys():
    evo_stats.append(fid_map[pathogen])

# bruteforce_stats = np.array(bruteforce_stats)
# evo_stats = np.array(evo_stats)
plt.scatter([1]*len(bruteforce_stats), bruteforce_stats, c='k', marker='o', label='bruteforce')
plt.scatter([2]*len(evo_stats),  evo_stats, c='r', marker='o', label='evolutionary')
plt.legend()

bruteforce_stats = np.array(bruteforce_stats)
evo_stats = np.array(evo_stats)
print('brutefroce: %.2f %.2f' % (np.mean(bruteforce_stats), np.std(bruteforce_stats)))
print('evo: %.2f %.2f' % (np.mean(evo_stats), np.std(evo_stats)))
print('t-test: %.2f p-val: %f' % ttest_ind(bruteforce_stats, evo_stats))
plt.show()