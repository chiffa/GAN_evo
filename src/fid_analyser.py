import os
import subprocess

balancing_folders_location = '/home/kucharav/FID_samples'
fid_command = 'home/kucharav/Documents/pytorch-fid-master/fid_score.py'

for random_tag in os.listdir(balancing_folders_location):
    print(random_tag)
    current_real = balancing_folders_location + random_tag + 'real'
    current_fake = balancing_folders_location + random_tag + 'fake'

    compled_process = subprocess.run([fid_command, current_real, current_fake])

    print(compled_process.stdout)

