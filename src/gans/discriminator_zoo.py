import torch
import torch.nn as nn
from src.gans.nn_structure import NetworkStructure
from random import sample
import string

char_set = string.ascii_uppercase + string.digits

class GaussianNoise(nn.Module):

    def __init__(self, sigma=0.1, device="cuda:1"):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = True
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Discriminator(nn.Module):

    def __init__(self, ngpu, latent_vector_size, discriminator_latent_maps, number_of_colors):
        super(Discriminator, self).__init__()
        self.tag = "disc_base"
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.ngpu = ngpu
        self.latent_vector_size = latent_vector_size
        self.discriminator_latent_maps = discriminator_latent_maps
        self.number_of_colors = number_of_colors
        self.encounter_trace = []
        # TODO: Gaussian noise injection
        # self.noise = GaussianNoise()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=self.number_of_colors,
                      out_channels=self.discriminator_latent_maps,
                      kernel_size=4,
                      stride=2,  # affects the size of the out map (divides)
                      padding=1,  # affects the size of the out map
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.discriminator_latent_maps, self.discriminator_latent_maps * 2, 4, 2, 1,
                      bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.discriminator_latent_maps * 2, self.discriminator_latent_maps * 4, 4, 2,
                      1,
                      bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.discriminator_latent_maps * 4, self.discriminator_latent_maps * 8, 4, 2,
                      1, bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.discriminator_latent_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def bind_nn_structure(self, network: NetworkStructure):
        #TODO: check if in/out dimensions are consistent
        self.main = nn.Sequential(network.compile())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            input = self.noise(input)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            input = self.noise(input)
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

    def save_instance_state(self):
        pass

    def recover_from_store(self, stored_state):
        pass

    def size_on_disc(self):
        return count_parameters(self.main)


class Discriminator_with_full_linear(nn.Module):

    def __init__(self, ngpu, latent_vector_size, discriminator_latent_maps, number_of_colors):
        super(Discriminator, self).__init__()
        self.tag = 'disc_with_linear'
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.ngpu = ngpu
        self.latent_vector_size = latent_vector_size
        self.discriminator_latent_maps = discriminator_latent_maps
        self.number_of_colors = number_of_colors
        # TODO: Gaussian noise injection
        # self.noise = GaussianNoise()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=self.number_of_colors,
                      out_channels=self.discriminator_latent_maps,
                      kernel_size=4,
                      stride=2,  # affects the size of the out map (divides)
                      padding=1,  # affects the size of the out map
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.discriminator_latent_maps, self.discriminator_latent_maps * 2, 4, 2, 1,
                      bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Linear((self.discriminator_latent_maps*2) * 16 * 16,
                      (self.discriminator_latent_maps*2) * 16 * 16),
            nn.BatchNorm2d(self.discriminator_latent_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.discriminator_latent_maps * 2, self.discriminator_latent_maps * 4, 4, 2,
                      1,
                      bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.discriminator_latent_maps * 4, self.discriminator_latent_maps * 8, 4, 2,
                      1, bias=False),
            nn.BatchNorm2d(self.discriminator_latent_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.discriminator_latent_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def bind_nn_structure(self, network: NetworkStructure):
        #TODO: check if in/out dimensions are consistent
        self.main = nn.Sequential(network.compile())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            input = self.noise(input)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            input = self.noise(input)
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

    def size_on_disc(self):
        return count_parameters(self.main)


class Discriminator_PReLU(nn.Module):

        def __init__(self, ngpu, latent_vector_size, discriminator_latent_maps, number_of_colors):
            super(Discriminator, self).__init__()
            self.tag = "disc_PReLU"
            self.random_tag = ''.join(sample(char_set * 10, 10))
            self.ngpu = ngpu
            self.latent_vector_size = latent_vector_size
            self.discriminator_latent_maps = discriminator_latent_maps
            self.number_of_colors = number_of_colors
            # TODO: Gaussian noise injection
            # self.noise = GaussianNoise()
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(in_channels=self.number_of_colors,
                          out_channels=self.discriminator_latent_maps,
                          kernel_size=4,
                          stride=2,  # affects the size of the out map (divides)
                          padding=1,  # affects the size of the out map
                          bias=False),
                nn.PReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(self.discriminator_latent_maps, self.discriminator_latent_maps * 2, 4, 2,
                          1,
                          bias=False),
                nn.BatchNorm2d(self.discriminator_latent_maps * 2),
                nn.PReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(self.discriminator_latent_maps * 2, self.discriminator_latent_maps * 4, 4,
                          2,
                          1,
                          bias=False),
                nn.BatchNorm2d(self.discriminator_latent_maps * 4),
                nn.PReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(self.discriminator_latent_maps * 4, self.discriminator_latent_maps * 8, 4,
                          2,
                          1, bias=False),
                nn.BatchNorm2d(self.discriminator_latent_maps * 8),
                nn.PReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(self.discriminator_latent_maps * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def bind_nn_structure(self, network: NetworkStructure):
            # TODO: check if in/out dimensions are consistent
            self.main = nn.Sequential(network.compile())

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                input = self.noise(input)
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                input = self.noise(input)
                output = self.main(input)

            return output.view(-1, 1).squeeze(1)

        def size_on_disc(self):
            return count_parameters(self.main)
