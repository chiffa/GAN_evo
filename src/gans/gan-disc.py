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


image_folder = ""
number_of_colors = 1
image_size = 64

memoized_discriminator = ""
memoized_generator = ""

memoization_location = ""

workers = 2
batch_size = 64
ngpu = 1

latent_vector_size = 64
generator_latent_maps = 64
discriminator_latent_maps = 64

learning_rate = 0.0002
beta1 = 0.5
training_epochs = 25

device = torch.device("cpu")


dataset = dset.MNIST(root=image_folder, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=int(workers))


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=latent_vector_size,
                               out_channels=generator_latent_maps * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=generator_latent_maps * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=generator_latent_maps * 8,
                               out_channels=generator_latent_maps * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(generator_latent_maps * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=generator_latent_maps * 4,
                               out_channels=generator_latent_maps * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(generator_latent_maps * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=generator_latent_maps * 2,
                               out_channels=generator_latent_maps,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(generator_latent_maps),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=generator_latent_maps,
                               out_channels=number_of_colors,
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

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=number_of_colors,
                      out_channels=discriminator_latent_maps,
                      kernel_size=4,
                      stride=2,  # affects the size of the out map (divides)
                      padding=1,  # affects the size of the out map
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(discriminator_latent_maps, discriminator_latent_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_latent_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(discriminator_latent_maps * 2, discriminator_latent_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_latent_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(discriminator_latent_maps * 4, discriminator_latent_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_latent_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(discriminator_latent_maps * 8, 1, 4, 1, 0, bias=False),
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


Generator_instance = Generator(ngpu).to(device)
Generator_instance.apply(weights_init)
if memoized_generator != '':
    Generator_instance.load_state_dict(torch.load(memoized_generator))
print(Generator_instance)

Discriminator_instance = Discriminator(ngpu).to(device)
Discriminator_instance.apply(weights_init)
if memoized_discriminator != '':
    Discriminator_instance.load_state_dict(torch.load(memoized_discriminator))
print(Discriminator_instance)

criterion = nn.BCELoss()

fixed_noise = torch.randn(batch_size,
                          latent_vector_size,
                          1,
                          1,
                          device=device)
real_label = 1
fake_label = 0

# TODO: this needs to be parametrized in the future
optimizerD = optim.Adam(Discriminator_instance.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(Generator_instance.parameters(), lr=learning_rate, betas=(beta1, 0.999))


def do_pair_training():
    for epoch in range(training_epochs):
        # that will go into arena
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            Discriminator_instance.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = Discriminator_instance(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
            fake = Generator_instance(noise)
            label.fill_(fake_label)
            output = Discriminator_instance(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            Generator_instance.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = Discriminator_instance(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, training_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % memoization_location,
                                  normalize=True)
                fake = Generator_instance(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d.png' % (memoization_location, epoch),
                                  normalize=True)

        # do checkpointing
        torch.save(Generator_instance.state_dict(), '%s/netG_epoch_%d.pth' % (memoization_location, epoch))
        torch.save(Discriminator_instance.state_dict(), '%s/netD_epoch_%d.pth' % (memoization_location, epoch))

if __name__ == "__main__":
    pass
