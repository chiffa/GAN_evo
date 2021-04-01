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
from src.gans.discriminator_zoo import Discriminator
from src.gans.generator_zoo import Generator
from os.path import abspath, join
from random import sample
import string
import numpy as np
from src.mongo_interface import save_pure_disc, save_pure_gen, filter_pure_disc, \
    filter_pure_gen, update_pure_disc, update_pure_gen
from src.scoring_models import pathogen_host_fitness, cumulative_host_fitness
from configs import cuda_device
from configs import training_samples_location as _train_samples_dir
from configs import fid_samples_location as _fid_samples_dir


#Environment to match and train our gans (can be either disc only or gen only or both)
#main test environment where the training of GANs is performed


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
                 ngpu=1, workers=2, device=cuda_device,
                 sample_image_folder=_train_samples_dir,
                 fid_image_folder=_fid_samples_dir,
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

        self.batch_size = batch_size

        self.latent_vector_size = latent_vector_size
        self.sample_image_folder = sample_image_folder
        self.fid_image_folder = fid_image_folder

        try:
            os.makedirs(self.sample_image_folder)
        except OSError:
            pass

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_training_round(generator_instance, discriminator_instance,
                         disc_optimizer, gen_optimizer, criterion,
                         dataloader, device, latent_vector_size, mode="match",
                         real_label=1, fake_label=0, training_epochs=1,
                         noise_floor=0.01, fitness_biases=(1, 1),
                         timer=None):

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
        for i, data in enumerate(dataloader, 0):

            if dataloader_limiter is not None and i > dataloader_limiter:
                break

            if timer is not None:
                timer.start()

            # train with real data
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

            if timer is not None:
                timer.stop()

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

            if match:
                match_trace.append([average_disc_error_on_real,
                                    average_disc_error_on_gan])
        print()

    # TODO: potential optimization, although not a very potent one.
    # matching requires no real data training.
    # training needs to return the average error on the reals, but can't - because that's
    # the last pass one that finishes the training, without any backward propagation

    if train_g or train_d:
        return np.array(training_trace).tolist()

    if match:
        match_trace = np.array(match_trace)
        match_trace = np.mean(match_trace, axis=0)
        return match_trace.tolist()


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

    def match(self, timer=None, commit=True):
        trace = match_training_round(self.generator_instance, self.discriminator_instance,
                                     self.discriminator_optimizer, self.generator_optimizer,
                                     self.criterion,
                                     self.env.dataloader, self.env.device,
                                     self.env.latent_vector_size,
                                     mode="match",
                                     real_label=self.env.true_label,
                                     fake_label=self.env.fake_label,
                                     training_epochs=1,
                                     timer=timer)

        d_encounter_trace = [type(self.generator_instance).__name__,
                             self.generator_instance.tag,
                             [],
                             trace]

        g_encounter_trace = [type(self.discriminator_instance).__name__,
                             self.discriminator_instance.tag,
                             [],
                             trace]

        self.discriminator_instance.encounter_trace.append(d_encounter_trace)
        self.generator_instance.encounter_trace.append(g_encounter_trace)

        #TODO: add the weigtings by autoimmunity and virulence

        # print('debug: inside match: real_error: %s, false_error: %s' % (trace[0], trace[1]))

        host_fitness, pathogen_fitness = pathogen_host_fitness(trace[0], trace[1])
        self.discriminator_instance.real_error = trace[0]

        # print('debug: inside match: host_fitness: %s, pathogen_fitness: %s' % (host_fitness,
        #                                                                        pathogen_fitness))

        # TODO: check for conflict with the decisions made inside the arena module
        if pathogen_fitness > 1:
            # print('debug: inside match: contamination branch')  # contamination
            self.generator_instance.fitness_map = {
                self.discriminator_instance.random_tag: pathogen_fitness}
            self.discriminator_instance.gen_error_map = {self.generator_instance.random_tag:
                                                             trace[1]}

        else:  # No contamination
            # print('debug: inside match: no-contamination branch')
            # clear pathogens if exist
            self.generator_instance.fitness_map.pop(self.discriminator_instance.random_tag, None)
            self.discriminator_instance.gen_error_map.pop(self.generator_instance.random_tag, None)

        
        self.discriminator_instance.current_fitness = cumulative_host_fitness(trace[0],
                                                                              self.discriminator_instance.gen_error_map.values())

        if commit:
            self.commit_disc_gen_updates()

        return trace

    def commit_disc_gen_updates(self):
            update_pure_disc(self.discriminator_instance.random_tag,
                             {'encounter_trace': self.discriminator_instance.encounter_trace,
                              'self_error': self.discriminator_instance.real_error,
                              'gen_error_map': self.discriminator_instance.gen_error_map,
                              'current_fitness': self.discriminator_instance.current_fitness})
            update_pure_gen(self.generator_instance.random_tag,
                            {'encounter_trace': self.generator_instance.encounter_trace,
                             'fitness_map': self.generator_instance.fitness_map})

    def first_disc_gen_commit(self, gen_only=False, disc_only=False):

        if not disc_only:
            save_pure_gen(self.generator_instance.save_instance_state())
        if not gen_only:
            save_pure_disc(self.discriminator_instance.save_instance_state())

    
    def cross_train(self, epochs=1, gan_only=False, disc_only=False, timer=None, commit=False):

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
                                     training_epochs=epochs,
                                     timer=timer)

        d_encounter_trace = [type(self.generator_instance).__name__,
                             self.generator_instance.tag,
                             trace, []]

        g_encounter_trace = [type(self.discriminator_instance).__name__,
                             self.discriminator_instance.tag,
                             trace, []]

        self.discriminator_instance.encounter_trace.append(d_encounter_trace)
        self.generator_instance.encounter_trace.append(g_encounter_trace)

        print('disc: ', self.discriminator_instance.random_tag, '->', end='')
        self.discriminator_instance.bump_random_tag()
        print(self.discriminator_instance.random_tag)
        print('gen: ', self.generator_instance.random_tag, '->', end='')
        self.generator_instance.bump_random_tag()
        print(self.generator_instance.random_tag)

        if commit:
            self.first_disc_gen_commit(disc_only=disc_only, gen_only=gan_only)

        return trace

    def sample_images(self, annotation=''):

        data = next(iter(self.env.dataloader))
        real_cpu = data[0].to(self.env.device)

        _batch_size = real_cpu.size(0)
        noise = torch.randn(self.env.batch_size, self.env.latent_vector_size, 1, 1,
                            device=self.env.device)
        fake = self.generator_instance(noise)

        name_correction = self.generator_instance.random_tag
        if len(annotation) > 0:
            name_correction = name_correction + '.' + annotation

        vutils.save_image(real_cpu,
                          '%s/%s.real_samples.png' % (self.env.sample_image_folder,
                                                      name_correction),
                          normalize=True)

        vutils.save_image(fake.detach(),
                          '%s/%s.fake_samples.png' % (self.env.sample_image_folder,
                                                      name_correction),
                          normalize=True)

        localized_fid_folder = os.path.join(self.env.fid_image_folder,
                                            self.generator_instance.random_tag)

        localized_fid_folder_f = os.path.join(localized_fid_folder, 'fake')
        localized_fid_folder_r = os.path.join(localized_fid_folder, 'real')

        try:
            os.makedirs(localized_fid_folder_f)
        except OSError:
            pass

        try:
            os.makedirs(localized_fid_folder_r)
        except OSError:
            pass

        for i in range(0, 64):
            vutils.save_image(real_cpu[i, :, :, :],
                              '%s/real_%d.png' % (localized_fid_folder_r, i),
                              normalize=True)

            vutils.save_image(fake.detach()[i, :, :, :],
                              '%s/fake_%d.png' % (localized_fid_folder_f, i),
                              normalize=True)


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

    # ========

    # gan_1 = mnist_gan_trainer.restore()
