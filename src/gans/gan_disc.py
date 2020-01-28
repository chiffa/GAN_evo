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
from src.mongo_interface import gan_pair_push_to_db, gan_pair_get_from_db, gan_pair_update_in_db
from os.path import abspath
from random import sample
import string
import numpy as np


char_set = string.ascii_uppercase + string.digits


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def margin_to_score_update():
    pass


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


class GanTrainer(object):

    def __init__(self, dataset,
                    ngpu=1, workers=2, batch_size=64,
                    latent_vector_size=64, generator_latent_maps=64, discriminator_latent_maps=64,
                    learning_rate=0.0002, beta1=0.5, training_epochs=25,
                    device="cuda:1", memoization_location="./memoized",
                    number_of_colors=1, image_dimensions=64, image_type='mnist',
                    from_dict=None):


        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.dataset_type = type(dataset).__name__

        self.disc_elo = 1500
        self.gen_elo = 1500
        self.matches = 0

        self.number_of_colors = number_of_colors
        self.memoization_location = memoization_location
        self.image_dimensions = image_dimensions
        self.image_type = image_type

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

        self.training_trace = []

        if from_dict is not None:
            self.random_tag = from_dict['random_tag']
            self.image_type, self.image_dimensions, dataset_name = from_dict['image_chars']
            self.workers, self.batch_size = from_dict['training_params']
            self.latent_vector_size, self.generator_latent_maps, self.discriminator_latent_maps =\
                from_dict['latent_maps_params']
            self.learning_rate, self.beta1, self.training_epochs, self.real_label, self.fake_label,\
                criterion_name, G_optimizer_name, D_optimizer_name = \
                from_dict['training_parameters']
            self.Discriminator_instance = Discriminator(ngpu,
                                                        self.latent_vector_size,
                                                        self.discriminator_latent_maps,
                                                        self.number_of_colors).to(device)
            self.Generator_instance = Generator(ngpu,
                                                self.latent_vector_size,
                                                self.generator_latent_maps,
                                                self.number_of_colors).to(self.device)
            self.Generator_instance.load_state_dict(from_dict['Generator_state'])
            self.Discriminator_instance.load_state_dict(from_dict['Discriminator_state'])
            self.matches, self.disc_elo, self.gen_elo = from_dict['score_ratings']
            self.training_trace = from_dict['training_trace']

            if self.dataset_type != dataset_name or \
                type(self.criterion).__name__ != criterion_name or \
                type(self.optimizerG).__name__ != G_optimizer_name or \
                type(self.optimizerD).__name__ != D_optimizer_name:
                raise Exception('Inconsistent names: '
                                '\n\tdataset: %s || %s'
                                '\n\tcriterion: %s || %s'
                                '\n\toptimizerG: %s || %s'
                                '\n\toptimizerD: %s || %s' %
                                (self.dataset_type, dataset_name,
                                 type(self.criterion).__name__, criterion_name,
                                 type(self.optimizerG).__name__, G_optimizer_name,
                                 type(self.optimizerD).__name__, D_optimizer_name))


    def retrieve_from_memoization(self, memoized_discriminator="", memoized_generator=""):

        if memoized_generator != '':
            self.Generator_instance.load_state_dict(torch.load(memoized_generator))

        if memoized_discriminator != '':
            self.Discriminator_instance.load_state_dict(torch.load(memoized_discriminator))

    def hyperparameters_key(self):
        key = {'random_tag': self.random_tag,
               'image_chars': (self.image_type,
                               self.image_dimensions,
                               self.dataset_type),
               'training_params': (self.workers,
                                   self.batch_size),
               'latent_maps_params': (self.latent_vector_size,
                                      self.generator_latent_maps,
                                      self.discriminator_latent_maps),
               'training_parameters': (self.learning_rate, self.beta1,
                                       self.training_epochs,
                                       self.real_label, self.fake_label,
                                       type(self.criterion).__name__,
                                       type(self.optimizerG).__name__,
                                       type(self.optimizerD).__name__)}
        return key

    def save(self):
        payload = {'Generator_state': self.Generator_instance.state_dict(),
                   'Discriminator_state': self.Discriminator_instance.state_dict(),
                   'score_ratings': (self.matches, self.disc_elo, self.gen_elo),
                   'training_trace': self.training_trace}
        payload.update(self.hyperparameters_key())

        gan_pair_push_to_db(payload)

    def restore(self):
        query_result = gan_pair_get_from_db(self.hyperparameters_key())

        if query_result is not None:
            self.Generator_instance.load_state_dict(query_result['Generator_state'])
            self.Discriminator_instance.load_state_dict(query_result['Discriminator_state'])
            self.matches, self.disc_elo, self.gen_elo = query_result['score_ratings']

    def update_match_results(self):
        gan_pair_update_in_db({'random_tag': self.random_tag},
                              {'score_ratings': (self.matches, self.disc_elo, self.gen_elo)})



    def do_pair_training(self, _epochs=None):

        if _epochs is not None:
            self.training_epochs = _epochs

        print('training %s with following parameter array: '
              'bs: %s, dlv: %s, glv: %s, '
              'lr: %.5f, b: %.2f, tep: %s' % (self.random_tag,
                                              self.batch_size, self.latent_vector_size,
                                              self.generator_latent_maps,
                                              self.learning_rate, self.beta1,
                                              self.training_epochs))

        for epoch in range(self.training_epochs):
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
                average_disc_success_on_real = output.mean().item()

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
                average_disc_error_on_gan_post_update = output.mean().item()
                self.optimizerG.step()

                print('[%02d/%02d][%03d/%03d]'
                      '\tdisc loss: %.4f; '
                      '\tgen loss: %.4f; '
                      '\tdisc success on real: %.4f; '
                      '\tdisc error on gen pre/post update: %.4f / %.4f; '
                      % (epoch, self.training_epochs, i, len(self.dataloader),
                         total_discriminator_error.item(),
                         errG.item(),
                         average_disc_success_on_real,
                         average_disc_error_on_gan,
                         average_disc_error_on_gan_post_update),
                      end='\r')

                self.training_trace.append([epoch, i,
                         total_discriminator_error.item(),
                         errG.item(),
                         average_disc_success_on_real,
                         average_disc_error_on_gan,
                         average_disc_error_on_gan_post_update])

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
            print('')


    def match(self, oponnent):

        if self.hyperparameters_key()['image_chars'] != \
            oponnent.hyperparameters_key()['image_chars']:
            raise(Exception('incompatible images are being compared: %s (self) vs %s' %
                            (self.hyperparameters_key()['image_chars'],
                             oponnent.hyperparameters_key()['image_chars'])))

        for i, data in enumerate(self.dataloader, 0):
            real_cpu = data[0].to(self.device)
            _batch_size = real_cpu.size(0)
            label = torch.full((_batch_size,), self.real_label, device=self.device)
            output = self.Discriminator_instance(real_cpu)
            self_errD_real = self.criterion(output, label)
            # self discriminator performance on real

            real_cpu = data[0].to(oponnent.device)
            _batch_size = real_cpu.size(0)
            label = torch.full((_batch_size,), oponnent.real_label, device=oponnent.device)
            output = oponnent.Discriminator_instance(real_cpu)
            oponnent_errD_real = oponnent.criterion(output, label)
            # opponnent discriminator performance on real data

            noise = torch.randn(_batch_size, self.latent_vector_size, 1, 1,
                                device=self.device)
            fake = self.Generator_instance(noise)
            label.fill_(self.fake_label)
            output = self.Discriminator_instance(fake.detach())
            self_gen_self_disc_av_err = output.mean().item()
            self_gen_self_disc_errD = self.criterion(output, label)
            # self discriminator performance on self fake

            noise = torch.randn(_batch_size, oponnent.latent_vector_size, 1, 1,
                                device=oponnent.device)
            fake = oponnent.Generator_instance(noise)
            label.fill_(oponnent.fake_label)
            output = oponnent.Discriminator_instance(fake.detach())
            opp_gen_opp_disc_av_err = output.mean().item()
            opp_gen_opp_disc_errD = oponnent.criterion(output, label)
            # oponnent performance on my oponnent's fake
            
            
            noise = torch.randn(_batch_size, oponnent.latent_vector_size, 1, 1,
                                device=oponnent.device)
            fake = oponnent.Generator_instance(noise)
            label.fill_(self.fake_label)
            output = self.Discriminator_instance(fake.detach())
            opp_gen_self_disc_av_err = output.mean().item()
            opp_gen_self_disc_errD = self.criterion(output, label)
            # self discriminator performance on opponent's fake

            noise = torch.randn(_batch_size, self.latent_vector_size, 1, 1,
                                device=self.device)
            fake = self.Generator_instance(noise)
            label.fill_(oponnent.fake_label)
            output = oponnent.Discriminator_instance(fake.detach())
            self_gen_opp_disc_av_err = output.mean().item()
            self_gen_opp_disc_errD = oponnent.criterion(output, label)
            # oponnent performance on my self's fake

            self_discriminator_error = self_errD_real + \
                                             self_gen_self_disc_errD + \
                                             opp_gen_self_disc_errD

            oponnent_discriminator_error = oponnent_errD_real + \
                                                opp_gen_opp_disc_errD + \
                                                self_gen_opp_disc_errD

            self_gan_performance = np.min([self_gen_opp_disc_av_err,
                                           self_gen_self_disc_av_err])

            oponnent_gan_performance = np.min([opp_gen_self_disc_av_err,
                                              opp_gen_opp_disc_av_err])


            disc_margin = ( -self_discriminator_error +
                           oponnent_discriminator_error).cpu().detach().float()
            gen_margin = self_gan_performance - oponnent_gan_performance

            # print(disc_margin)
            # print(gen_margin)

            # TODO: ugly hacks from here on

            if self.disc_elo == oponnent.disc_elo:
                self.disc_elo -= 5
                oponnent.disc_elo += 5

            if self.gen_elo == oponnent.gen_elo:
                self.gen_elo -= 5
                oponnent.gen_elo+=5

            if disc_margin > 0:  # my disc won
                margin_multiplier = np.log(abs(disc_margin) + 1) * (2.2 /(self.disc_elo -
                                                                     oponnent.disc_elo)*0.001+2.2)
                self.disc_elo += margin_multiplier/2
                oponnent.disc_elo -= margin_multiplier/2

            elif disc_margin < 0:  # oponnent's disc won
                margin_multiplier = np.log(abs(disc_margin) + 1) * (2.2 / (oponnent.disc_elo -
                                                                      self.disc_elo) * 0.001 + 2.2)
                oponnent.disc_elo += margin_multiplier / 2
                self.disc_elo -= margin_multiplier / 2

            if gen_margin > 0:  # my gen won
                margin_multiplier = np.log(abs(gen_margin) + 1) * (2.2 /(self.gen_elo -
                                                                     oponnent.gen_elo)*0.001+2.2)
                self.gen_elo += margin_multiplier/2
                oponnent.gen_elo -= margin_multiplier/2

            elif gen_margin < 0:  # oponnent's gen won
                margin_multiplier = np.log(abs(gen_margin) + 1) * (2.2 / (oponnent.gen_elo -
                                                                      self.gen_elo) * 0.001 + 2.2)
                oponnent.gen_elo += margin_multiplier / 2
                self.gen_elo -= margin_multiplier / 2

            print("\tself disc/gen perf: %.4f/%4.f;"
                  "\toppo disc/gen perf: %.4f/%.4f;"
                  "\tupdated disc elo scores: self/opp:%.2f/%.2f\t"
                  "\tupdated gen elo scores: self/opp:%.2f/%.2f" %
                  (self_discriminator_error.cpu().detach().float(),
                   self_gan_performance,
                   oponnent_discriminator_error, oponnent_gan_performance,
                   self.disc_elo, oponnent.disc_elo,
                   self.gen_elo, oponnent.gen_elo),
                  end='\r')

            self.matches += 1
            oponnent.matches += 1

        self.disc_elo = float(self.disc_elo)
        self.gen_elo = float(self.gen_elo)

        oponnent.disc_elo = float(oponnent.disc_elo)
        oponnent.gen_elo = float(oponnent.gen_elo)

        self.update_match_results()
        oponnent.update_match_results()
        print('\n')


if __name__ == "__main__":
    image_folder = "./image"
    image_size = 64
    number_of_colors = 1
    imtype = 'mnist'

    mnist_dataset = dset.MNIST(root=image_folder, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))

    mnist_gan_trainer = GanTrainer(mnist_dataset,
                                   number_of_colors=number_of_colors,
                                   image_dimensions=image_size,
                                   image_type=imtype,
                                   training_epochs=15)

    print(mnist_gan_trainer.random_tag)

    mnist_gan_trainer.do_pair_training()
    mnist_gan_trainer.save()

    # ========

    # gan_1 = mnist_gan_trainer.restore()
