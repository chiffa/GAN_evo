import os
import sys
from random import shuffle
from itertools import combinations
from src.fid_calc.fid_score import calculate_fid_given_paths
import pickle
import datetime
from configs import fid_samples_location

balancing_folders_location = fid_samples_location
# fid_command = '/home/kucharav/Documents/pytorch-fid-master/fid_score.py'
after_datetime = datetime.datetime.now() - datetime.timedelta(days=1)

#AMIR: diff between the 3 functions? added value from calculate_fid_given_path ..?

def calc_single_fid(random_tag):
    total_path = os.path.join(balancing_folders_location, random_tag)

    if os.path.isdir(total_path):
        current_real = balancing_folders_location + '/' + random_tag + '/' + 'real'
        current_fake = balancing_folders_location + '/' + random_tag + '/' + 'fake'

        try:
            fid_value = calculate_fid_given_paths([current_real, current_fake], 64, True, 2048)
            print('fid compute', random_tag, ': ', fid_value)
            return fid_value

        except:
            print("Unexpected error:", sys.exc_info()[0])

    return -1


def calc_gen_fids():
    random_tag_list = []
    fid_map = {}
    for random_tag in os.listdir(balancing_folders_location):
        if os.path.getmtime(random_tag) > after_datetime:
            random_tag_list.append(random_tag)
            current_real = balancing_folders_location + '/' + random_tag + '/' + 'real'
            current_fake = balancing_folders_location + '/' + random_tag + '/' + 'fake'

            try:
                fid_value = calculate_fid_given_paths([current_real, current_fake], 64, True, 2048)
                fid_map[random_tag] = fid_value
                print(random_tag, ': ', fid_value)
            except:
                print("Unexpected error:", sys.exc_info()[0])

    return fid_map, random_tag_list


def calc_reals_fid(random_tag_list):
    real_comparison = []
    shuffle(random_tag_list)
    blocker = 20
    for i, (random_tag_1, random_tag_2) in enumerate(combinations(random_tag_list, 2)):
        if i > blocker:
            break
        current_1 = balancing_folders_location + '/' + random_tag_1 + '/' + 'real'
        current_2 = balancing_folders_location + '/' + random_tag_2 + '/' + 'real'

        try:
            fid_value = calculate_fid_given_paths([current_1, current_2], 64, True, 2048)
            real_comparison.append(fid_value)
            print('real to real sample', ': ', fid_value)
        except:
            print("Unexpected error:", sys.exc_info()[0])

    return real_comparison


if __name__ == "__main__":
    fid_map, random_tag_list = calc_gen_fids()
    real_comparison = calc_reals_fid(random_tag_list)

    if os.path.isfile('fid_scores.dmp'):
        old_fid_map, old_real_comparison = pickle.load((fid_map,
                                                        real_comparison),
                                                       open('fid_scores.dmp', 'rb'))
        fid_map.update(old_fid_map)
        real_comparison += old_real_comparison

    pickle.dump((fid_map, real_comparison), open('fid_scores.dmp', 'wb'))
