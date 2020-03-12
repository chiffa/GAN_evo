import os
from random import shuffle
from itertools import combinations
from src.fid_calc.fid_score import calculate_fid_given_paths
import pickle

balancing_folders_location = '/home/kucharav/FID_samples'
fid_command = '/home/kucharav/Documents/pytorch-fid-master/fid_score.py'

random_tag_list = []
fid_map = {}
real_comparison = []

for random_tag in os.listdir(balancing_folders_location):
    random_tag_list.append(random_tag)
    current_real = balancing_folders_location + '/' + random_tag + '/' + 'real'
    current_fake = balancing_folders_location + '/' + random_tag + '/' + 'fake'

    fid_value = calculate_fid_given_paths([current_real, current_fake], 64, True, 2048)
    fid_map[random_tag] = fid_value
    print(random_tag, ': ', fid_value)


shuffle(random_tag_list)


for random_tag_1, random_tag_2 in combinations(random_tag_list, 2):
    current_1 = balancing_folders_location + '/' + random_tag_1 + '/' + 'real'
    current_2 = balancing_folders_location + '/' + random_tag_2 + '/' + 'real'

    fid_value = calculate_fid_given_paths([current_1, current_2], 64, True, 2048)
    real_comparison.append(fid_value)
    print('real to real sample', ': ', fid_value)


pickle.dump((fid_map, real_comparison), open('fid_scores.dmp', 'w'))