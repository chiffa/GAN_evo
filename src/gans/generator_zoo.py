import torch.nn as nn
from src.gans.nn_structure import NetworkStructure


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Generator(nn.Module):

    def __init__(self, ngpu, latent_vector_size, generator_latent_maps, number_of_colors):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.latent_vector_size = latent_vector_size
        self.generator_latent_maps = generator_latent_maps
        self.number_of_colors = number_of_colors
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

