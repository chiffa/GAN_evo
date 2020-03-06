import torch.nn as nn
from src.gans.nn_structure import NetworkStructure
from random import sample
import string
import pickle
import torchvision.utils as vutils

char_set = string.ascii_uppercase + string.digits


def generate_hyperparameter_key(_self):
    key = {'random_tag': _self.random_tag,
           'gen_type': type(_self).__name__,
           'gen_latent_params': _self.gnenerator_latent_maps}
    return key


def save(_self):
    key = _self.generate_hyperparameter_key()
    payload = {'encounter_trace': _self.encounter_trace,
               'gen_state': pickle.dumps(_self.state_dict()),
               'fitness_map': _self.fitness_map}

    key.update(payload)

    return key


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Generator(nn.Module):


    # TODO: change to the environment binding
    def __init__(self, ngpu, latent_vector_size, generator_latent_maps, number_of_colors,
                 virulence = 20):
        super(Generator, self).__init__()
        self.tag = "gen_base"
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.ngpu = ngpu
        self.latent_vector_size = latent_vector_size
        self.generator_latent_maps = generator_latent_maps
        self.number_of_colors = number_of_colors
        self.fitness_map = {}
        self.encounter_trace = []
        self.tag_trace = [self.random_tag]
        self.virulence = virulence
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=self.latent_vector_size,
                               out_channels=self.generator_latent_maps * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=self.generator_latent_maps * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=self.generator_latent_maps * 8,
                               out_channels=self.generator_latent_maps * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self.generator_latent_maps * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=self.generator_latent_maps * 4,
                               out_channels=self.generator_latent_maps * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self.generator_latent_maps * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=self.generator_latent_maps * 2,
                               out_channels=self.generator_latent_maps,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self.generator_latent_maps),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=self.generator_latent_maps,
                               out_channels=self.number_of_colors,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def bind_nn_structure(self, network: NetworkStructure):
        # TODO: check that the in/out dimensions are consistent
        self.main = nn.Sequential(network.compile())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

    def size_on_disc(self):
        return count_parameters(self.main)

    def generate_hyperparameter_key(self):
        return generate_hyperparameter_key(self)

    def save_instance_state(self):
        return save(self)

    def bump_random_label(self):
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.tag_trace += [self.random_tag]
