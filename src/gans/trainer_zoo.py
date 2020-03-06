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
from src.gans.discriminator_zoo import Discriminator
from src.gans.generator_zoo import Generator
from os.path import abspath, join
from random import sample
import string
import numpy as np
from src.new_mongo_interface import save_pure_disc, save_pure_gen, filter_pure_disc, \
    filter_pure_gen, update_pure_disc, update_pure_gen
from src.scoring_models import pathogen_host_fitness, cumulative_host_fitness


char_set = string.ascii_uppercase + string.digits


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GANEnvironment(object):

    def __init__(self, dataset,
                 number_of_colors=1, image_dimensions=64, batch_size=64,
                 ngpu=1, workers=2, device="cuda:1",
                 sample_image_folder='/home/kucharav/trainer_samples',
                 true_label=1, fake_label=0, latent_vector_size=64):

        self.number_of_colors = number_of_colors
        self.image_dimensions = image_dimensions
        self.image_type = type(dataset).__name__

        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=int(workers))
        self.dataset_type = type(dataset).__name__

        self.device = torch.device(device)
        self.ngpu = ngpu
        self.true_label = true_label
        self.fake_label = fake_label

        self.latent_vector_size = latent_vector_size
        self.sample_image_folder = sample_image_folder

    def hyperparameter_key(self):
        key = {'image_params': (self.image_type,
                               self.image_dimensions,
                               self.dataset_type,
                               self.number_of_colors),
               'labeling_params': (self.true_label,
                                  self.fake_label),
               'env_latent_params': self.latent_vector_size
               }
        return key


def margin_to_score_update():
    pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_training_round(generator_instance, discriminator_instance,
                         disc_optimizer, gen_optimizer, criterion,
                         dataloader, device, latent_vector_size, mode="match",
                         real_label=1, fake_label=0, training_epochs=1,
                         noise_floor=0.01, fitness_biases=(1, 1)):

    training_trace = []
    match_trace = []

    match = False
    train_d = False
    train_g = False

    if mode == "match":
        match = True

    if mode == "train":
        train_d = True
        train_g = True

    if mode == "train_g":
        train_g = True

    if mode == "train_d":
        train_d = True

    dataloader_limiter = None

    if training_epochs < 1:
        dataloader_limiter = int(len(dataloader)*training_epochs)
        training_epochs = 1


    for epoch in range(training_epochs):
        for i, data in enumerate(dataloader, 0)[:dataloader_limiter]:

            # train with real
            discriminator_instance.zero_grad()
            real_cpu = data[0].to(device)
            _batch_size = real_cpu.size(0)
            label = torch.full((_batch_size,), real_label, device=device)

            output = discriminator_instance(real_cpu)
            errD_real = criterion(output, label)

            if train_d:
                errD_real.backward()
            average_disc_success_on_real = output.mean().item()

            # train with fake
            noise = torch.randn(_batch_size, latent_vector_size, 1, 1, device=device)
            fake = generator_instance(noise)  # generates fake data

            label.fill_(fake_label)
            output = discriminator_instance(fake.detach())  # flags input as
            # non-gradientable
            errD_fake = criterion(output, label)  # calculates the loss for the prediction
            # error

            if train_d:
                errD_fake.backward()  # backpropagates it

            average_disc_error_on_gan = output.mean().item()
            average_disc_error_on_real = 1 - average_disc_success_on_real

            total_discriminator_error = errD_real + errD_fake

            if train_d:
                disc_optimizer.step()

            if train_g:
                generator_instance.zero_grad()  # clears gradients from the previous back
                label.fill_(real_label)  # fake labels are real for generator_instance cost
                output = discriminator_instance(fake)
                total_generator_error = criterion(output, label)
                total_generator_error.backward()
                average_disc_error_on_gan_post_update = output.mean().item()
                gen_optimizer.step()

            if train_g or train_d:

                if not train_g:
                    total_generator_error = total_discriminator_error
                    average_disc_error_on_gan_post_update = average_disc_error_on_gan

                print('[%02d/%02d][%03d/%03d]'
                      '\tdisc loss: %.4f; '
                      '\tgen loss: %.4f; '
                      '\tdisc success on real: %.4f; '
                      '\tdisc error on gen pre/post update: %.4f / %.4f; '
                      % (epoch, training_epochs, i, len(dataloader),
                         total_discriminator_error.item(),
                         total_generator_error.item(),
                         average_disc_success_on_real,
                         average_disc_error_on_gan,
                         average_disc_error_on_gan_post_update),
                      end='\r')

                training_trace.append([epoch, i,
                                            total_discriminator_error.item(),
                                            total_generator_error.item(),
                                            average_disc_success_on_real,
                                            average_disc_error_on_gan,
                                            average_disc_error_on_gan_post_update])
                return training_trace

            if match:
                match_trace.append([average_disc_error_on_real,
                                    average_disc_error_on_gan])

    # TODO: potential optimization, although not a very potent one.
    # matching requires no real data training.
    # training needs to return the average error on the reals, but can't - because that's
    # the last pass one that finishes the training, without any backward propagation

    if train_g or train_d:
        return np.array(train_g)

    if match:
        match_trace = np.array(match_trace)
        match_trace = np.mean(match_trace, axis=0)
        return match_trace



class Arena(object):

    def __init__(self, environment, generator_instance, discriminator_instance,
                 generator_optimizer_partial, discriminator_optimizer_partial,
                 criterion=nn.BCELoss()):

        self.env = environment

        self.generator_instance = generator_instance
        self.discriminator_instance = discriminator_instance

        self.generator_optimizer = generator_optimizer_partial(generator_instance.parameters())
        self.discriminator_optimizer = discriminator_optimizer_partial(
            discriminator_instance.parameters())

        self.criterion = criterion

    def decide_survival(self):

        pass

    def match(self):
        trace = match_training_round(self.generator_instance, self.discriminator_instance,
                             self.discriminator_optimizer, self.generator_optimizer,
                             self.criterion,
                             self.env.dataloader, self.env.device,
                             self.env.latent_vector_size,
                             mode="match",
                             real_label=self.env.true_label,
                             fake_label=self.env.fake_label,
                             training_epochs=1)

        d_encounter_trace = [type(self.generator_instance).__name__, self.generator_instance.tag,
                           [], trace]

        g_encounter_trace = [type(self.discriminator_instance).__name__,
                             self.discriminator_instance.tag,
                           [], trace]

        self.discriminator_instance.encounter_trace.append(d_encounter_trace)
        self.generator_instance.encounter_trace.append(g_encounter_trace)

        #TODO: add the weigtings by autoimmunity and virulence

        host_fitness, pathogen_fitness = pathogen_host_fitness(trace[0], trace[1])

        if pathogen_fitness > 1:  # contamination
            self.generator_instance.fitness_map = {
                self.discriminator_instance.random_tag: pathogen_fitness}
            self.discriminator_instance.gen_error_map = {self.generator_instance.radom_tag: trace[1]}


        else:  # No contamination
            # clear pathogens if exist
            self.generator_instance.fitness_map.pop(self.discriminator_instance.random_tag, None)
            self.discriminator_instance.gen_error_map.pop(self.generator_instance.radom_tag, None)

        self.discriminator_instance.current_fitness = cumulative_host_fitness(trace[0],
                                                                              self.generator_instance.fitness_map.values())

        update_pure_disc(self.discriminator_instance.random_tag,
                         {'encounter_trace': self.discriminator_instance.encounter_trace,
                          'self_error': trace[0],
                          'gen_error_map': self.discriminator_instance.fitness_map,
                          'current_fitness': self.discriminator_instance.current_fitness})

        update_pure_gen(self.generator_instance.random_tag,
                        {'encounter_trace': self.generator_instance.encounter_trace,
                         'fitness_map': self.generator_instance.fitness_map})



    def cross_train(self, epochs=1, gan_only=False, disc_only=False):

        mode = "train"

        if gan_only and disc_only:
            raise Exception('Both Gan and Disc training are set to only')
        if gan_only:
            mode = "train_g"
        if disc_only:
            mode = "train_d"

        trace = match_training_round(self.generator_instance, self.discriminator_instance,
                             self.discriminator_optimizer, self.generator_optimizer,
                             self.criterion,
                             self.env.dataloader, self.env.device,
                             self.env.latent_vector_size,
                             mode=mode,
                             real_label=self.env.true_label,
                             fake_label=self.env.fake_label,
                             training_epochs=epochs)

        d_encounter_trace = [type(self.generator_instance).__name__,
                             self.generator_instance.tag,
                             trace, []]

        g_encounter_trace = [type(self.discriminator_instance).__name__,
                             self.discriminator_instance.tag,
                             trace, []]

        self.discriminator_instance.encounter_trace.append(d_encounter_trace)
        self.generator_instance.encounter_trace.append(g_encounter_trace)

        self.discriminator_instance.bump_random_tag()
        self.generator_instance.bump_random_tag()

        save_pure_disc(self.discriminator_instance.save_insance_state)

        save_pure_gen(self.generator_instance.tag.save_instance_state)



class GanTrainer(object):

    def __init__(self, dataset,
                 ngpu=1, workers=2, batch_size=64,
                 latent_vector_size=64, generator_latent_maps=64, discriminator_latent_maps=64,
                 learning_rate=0.0002, beta1=0.5, training_epochs=25,
                 device="cuda:1", memoization_location="./memoized",
                 number_of_colors=1, image_dimensions=64, image_type='mnist',
                 from_dict=None, sample_image_folder='~/trainer_samples', size_hard_limiter=50000):

        self.size_hard_limiter = size_hard_limiter
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.dataset_type = type(dataset).__name__

        self.disc_elo = 1500
        self.gen_elo = 1500
        self.matches = 0

        self.number_of_colors = number_of_colors
        self.memoization_location = memoization_location
        self.image_dimensions = image_dimensions
        self.image_type = image_type

        # print(abspath(memoization_location))

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
                                '\n\tgen_optimizer: %s || %s'
                                '\n\tdisc_optimizer: %s || %s' %
                                (self.dataset_type, dataset_name,
                                 type(self.criterion).__name__, criterion_name,
                                 type(self.optimizerG).__name__, G_optimizer_name,
                                 type(self.optimizerD).__name__, D_optimizer_name))

        self.image_path = abspath(sample_image_folder)
        os.makedirs(self.image_path, exist_ok=True)


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
        if self.Generator_instance.size_on_disc > self.size_hard_limiter or \
            self.Discriminator_instance.size_on_disc > self.size_hard_limiter:
            print("Gen/disc model too big to store - saving dropped")
            return None

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

        if self.Generator_instance.size_on_disc > self.size_hard_limiter or \
            self.Discriminator_instance.size_on_disc > self.size_hard_limiter:
            print("Gen/disc model too big to store - training dropped")
            return None

        if _epochs is not None:
            self.training_epochs = _epochs

        # print('training %s with following parameter array: '
        #       'bs: %s, dlv: %s, glv: %s, '
        #       'lr: %.5f, b: %.2f, tep: %s' % (self.random_tag,
        #                                       self.batch_size, self.latent_vector_size,
        #                                       self.generator_latent_maps,
        #                                       self.learning_rate, self.beta1,
        #                                       self.training_epochs))

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
                label.fill_(self.real_label)  # fake labels are real for generator_instance cost
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
            # self discriminator_instance performance on real

            real_cpu = data[0].to(oponnent.device)
            _batch_size = real_cpu.size(0)
            label = torch.full((_batch_size,), oponnent.real_label, device=oponnent.device)
            output = oponnent.Discriminator_instance(real_cpu)
            oponnent_errD_real = oponnent.criterion(output, label)
            # opponnent discriminator_instance performance on real data

            noise = torch.randn(_batch_size, self.latent_vector_size, 1, 1,
                                device=self.device)
            fake = self.Generator_instance(noise)
            label.fill_(self.fake_label)
            output = self.Discriminator_instance(fake.detach())
            self_gen_self_disc_av_err = output.mean().item()
            self_gen_self_disc_errD = self.criterion(output, label)
            # self discriminator_instance performance on self fake

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
            # self discriminator_instance performance on opponent's fake

            noise = torch.randn(_batch_size, self.latent_vector_size, 1, 1,
                                device=self.device)
            fake = self.Generator_instance(noise)
            label.fill_(oponnent.fake_label)
            output = oponnent.Discriminator_instance(fake.detach())
            self_gen_opp_disc_av_err = output.mean().item()
            self_gen_opp_disc_errD = oponnent.criterion(output, label)
            # oponnent performance on my self's fake

            # TODO: move away from cross-entropyto the error on the relevant factors in %

            self_discriminator_error = self_errD_real + \
                                             self_gen_self_disc_errD + \
                                             opp_gen_self_disc_errD

            oponnent_discriminator_error = oponnent_errD_real + \
                                                opp_gen_opp_disc_errD + \
                                                self_gen_opp_disc_errD


            self_gan_performance = min(self_gen_opp_disc_av_err,
                                           self_gen_self_disc_av_err)

            oponnent_gan_performance = min(opp_gen_self_disc_av_err,
                                              opp_gen_opp_disc_av_err)


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
                oponnent.gen_elo += 5

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

            print("\tself disc err/gen perf: %.2E/%.2E;"
                  "\toppo disc err/gen perf: %.2E/%.2E;"
                  "\tupdated disc elo scores: self/opp:%.2f/%.2f\t"
                  "\tupdated gen elo scores: self/opp:%.2f/%.2f" %
                  (self_discriminator_error.cpu().detach().float(),
                   self_gan_performance,
                   oponnent_discriminator_error,
                   oponnent_gan_performance,
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


    def sample_images(self, annotation=''):

        data = next(iter(self.dataloader))
        real_cpu = data[0].to(self.device)

        _batch_size = real_cpu.size(0)
        noise = torch.randn(self.batch_size, self.latent_vector_size, 1, 1,
                            device=self.device)
        fake = self.Generator_instance(noise)

        name_correction = self.random_tag
        if len(annotation) > 0:
            name_correction = name_correction + '.' + annotation

        vutils.save_image(real_cpu,
                          '%s/%s.real_samples.png' % (self.image_path, name_correction),
                          normalize=True)

        vutils.save_image(fake.detach(),
                          '%s/%s.fake_samples.png' % (self.image_path, name_correction),
                          normalize=True)


def train_run(generator_instance, discriminator_instance,
              generator_optimizer, discriminator_optimizer,
              loss_criterion_function):
    """
    optimizer should be a partial function, mostly likely provided by the training
    instance, that only requires the parameters of the disc/generator_instance insances
    """

    pass


if __name__ == "__main__":
    image_folder = "./image"
    image_size = 64
    number_of_colors = 1
    imtype = 'mnist'

    image_samples_folder = '~/Trainer_samples'

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
