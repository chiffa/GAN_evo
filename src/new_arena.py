from src.gans.discriminator_zoo import Discriminator, Discriminator_PReLU, Discriminator_with_full_linear
from src.gans.generator_zoo import Generator
from src.gans.trainer_zoo import Arena, GANEnvironment
from src.new_mongo_interface import save_pure_disc, save_pure_gen, filter_pure_disc, filter_pure_gen

import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import combinations

