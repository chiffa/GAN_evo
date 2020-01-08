import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from src.gans.nn_structure import NetworkStructure
from os.path import abspath




#
# image_folder = "./images"
# number_of_colors = 1
# image_size = 64
#
# memoized_discriminator = ""  # just the path towards the memoization location
# memoized_generator = ""  # just the path towars the memoized location
#
# memoization_location = "./memoized"
# print(abspath(memoization_location))
#
#
# # Makes a directory where things are dumped.
# try:
#     os.makedirs(memoization_location)
# except OSError:
#     pass
#
#
# workers = 2
# batch_size = 64
# ngpu = 1
#
# latent_vector_size = 64
# generator_latent_maps = 64
# discriminator_latent_maps = 64
#
# learning_rate = 0.0002
# beta1 = 0.5
# training_epochs = 25
#
#
# # device = torch.device("cpu")
# device = torch.device("cuda:1")
#
#
# dataset = dset.MNIST(root=image_folder, download=True,
#                            transform=transforms.Compose([
#                                transforms.Resize(image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5,), (0.5,)),
#                            ]))
#
#
# dataloader = torch.utils.data.DataLoader(dataset,
#                                          batch_size=batch_size,
#                                          shuffle=True,
#                                          num_workers=int(workers))


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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


class Discriminator(nn.Module):

    def __init__(self, ngpu, latent_vector_size, discriminator_latent_maps, number_of_colors):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.latent_vector_size = latent_vector_size
        self.discriminator_latent_maps = discriminator_latent_maps
        self.number_of_colors = number_of_colors
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
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class Gan_Trainer(object):

    def __init__(self, dataset,
                    ngpu=1, workers=2, batch_size=64,
                    latent_vector_size=64, generator_latent_maps=64, discriminator_latent_maps=64,
                    learning_rate=0.0002, beta1=0.5, training_epochs=25,
                    device="cuda:1", memoization_location="./memoized",
                    number_of_colors=1, image_dimensions=64):

        self.number_of_colors = number_of_colors
        self.memoization_location = memoization_location
        self.image_dimensions = image_dimensions


        print(abspath(memoization_location))

        # Makes a directory where things are dumped.
        try:
            os.makedirs(memoization_location)
        except OSError:
            pass

        self.workers = workers
        self.batch_size = batch_size

        self.latent_vector_size = latent_vector_size
        self.generator_latent_maps = generator_latent_maps
        self.discriminator_latent_maps = discriminator_latent_maps


        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.training_epochs = training_epochs

        self.device = torch.device(device)
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=int(workers))

        self.Generator_instance = Generator(ngpu,
                                            self.latent_vector_size,
                                            self.generator_latent_maps,
                                            self.number_of_colors).to(self.device)
        self.Generator_instance.apply(weights_init)

        self.Discriminator_instance = Discriminator(ngpu,
                                                    self.latent_vector_size,
                                                    self.discriminator_latent_maps,
                                                    self.number_of_colors).to(device)
        self.Discriminator_instance.apply(weights_init)

        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(batch_size,
                                  latent_vector_size,
                                  1,
                                  1,
                                  device=device)

        self.real_label = 1
        self.fake_label = 0

        self.optimizerD = optim.Adam(self.Discriminator_instance.parameters(),
                                     lr=learning_rate, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.Generator_instance.parameters(),
                                                  lr=learning_rate, betas=(beta1, 0.999))


    def retrieve_from_memoization(self, memoized_discriminator="", memoized_generator=""):

        if memoized_generator != '':
            self.Generator_instance.load_state_dict(torch.load(memoized_generator))

        if memoized_discriminator != '':
            self.Discriminator_instance.load_state_dict(torch.load(memoized_discriminator))

    def do_pair_training(self):

        # # initialize the generator
        # Generator_instance = Generator(ngpu).to(device)
        # Generator_instance.apply(weights_init)
        # if memoized_generator != '':
        #     Generator_instance.load_state_dict(torch.load(memoized_generator))
        # print(Generator_instance)
        #
        # # initialize the discriminator
        # Discriminator_instance = Discriminator(ngpu).to(device)
        # Discriminator_instance.apply(weights_init)
        # if memoized_discriminator != '':
        #     Discriminator_instance.load_state_dict(torch.load(memoized_discriminator))
        # print(Discriminator_instance)
        #
        # # set the minmax game criterion
        # criterion = nn.BCELoss()
        #
        # fixed_noise = torch.randn(batch_size,
        #                           latent_vector_size,
        #                           1,
        #                           1,
        #                           device=device)
        # real_label = 1
        # fake_label = 0
        #
        # # set the optimizers
        # optimizerD = optim.Adam(Discriminator_instance.parameters(), lr=learning_rate,
        #                         betas=(beta1, 0.999))
        # optimizerG = optim.Adam(Generator_instance.parameters(), lr=learning_rate,
        #                         betas=(beta1, 0.999))

        for epoch in range(self.training_epochs):
            # that will go into arena
            for i, data in enumerate(self.dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                # train with real
                self.Discriminator_instance.zero_grad()
                real_cpu = data[0].to(self.device)
                _batch_size = real_cpu.size(0)
                label = torch.full((_batch_size,), self.real_label, device=self.device)

                output = self.Discriminator_instance(real_cpu)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(_batch_size, self.latent_vector_size, 1, 1, device=self.device)
                fake = self.Generator_instance(noise)    # generates fake data

                label.fill_(self.fake_label)
                output = self.Discriminator_instance(fake.detach())  # flags input as
                # non-gradientable
                errD_fake = self.criterion(output, label)  # calculates the loss for the prediction
                # error
                errD_fake.backward()    #backpropagates it

                average_disc_error_on_gan = output.mean().item()
                total_discriminator_error = errD_real + errD_fake
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.Generator_instance.zero_grad()  # clears gradients from the previous back
                # propagations
                label.fill_(self.real_label)  # fake labels are real for generator cost
                output = self.Discriminator_instance(fake)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, self.training_epochs, i, len(self.dataloader),
                         total_discriminator_error.item(), errG.item(), D_x, average_disc_error_on_gan, D_G_z2))

                if i % 100 == 0:  # that's a bit of a bruteforce for logging.
                    vutils.save_image(real_cpu,
                                      '%s/real_samples.png' % self.memoization_location,
                                      normalize=True)

                    fake = self.Generator_instance(self.fixed_noise)
                    vutils.save_image(fake.detach(),
                                      '%s/fake_samples_epoch_%03d.png' % (
                                          self.memoization_location, epoch),
                                      normalize=True)

            # do checkpointing
            torch.save(self.Generator_instance.state_dict(), '%s/netG_epoch_%d.pth' % (
                self.memoization_location, epoch))
            torch.save(self.Discriminator_instance.state_dict(), '%s/netD_epoch_%d.pth' % (
                self.memoization_location, epoch))


if __name__ == "__main__":
    image_folder = "./image"
    image_size = 64
    number_of_colors = 1

    mnist_dataset = dset.MNIST(root=image_folder, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))

    mnist_gan_trainer = Gan_Trainer(mnist_dataset,
                                    number_of_colors=number_of_colors,
                                    image_dimensions=image_size)
    mnist_gan_trainer.do_pair_training()
