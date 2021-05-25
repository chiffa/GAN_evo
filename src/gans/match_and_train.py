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


from src.evo_helpers import aneuploidization, log_normal_aneuploidization, dump_evo

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
    """
    A wrapper object that contains the configurations for the environment in which the GAN is
    trained
    """

    def __init__(self, dataset,
                 number_of_colors=1,
                 image_dimensions=64,
                 batch_size=64,
                 ngpu=1,
                 workers=2,
                 device=cuda_device,
                 sample_image_folder=_train_samples_dir,
                 fid_image_folder=_fid_samples_dir,
                 true_label=1,  # TRACING: from configs
                 fake_label=0,  # TRACING: from configs
                 latent_vector_size=64):

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
        except OSError:  # the folder to store the sample images exists
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


# CURRENTPASS: [complexity] cyclomatic complexity=20
#  Split the training round with the matching round
def match_training_round(generator_instance, discriminator_instance,
                         disc_optimizer, gen_optimizer, criterion,
                         dataloader, device, latent_vector_size, mode="match",
                         real_label=1, fake_label=0, training_epochs=1,
                         noise_floor=0.01, fitness_biases=(1, 1),                    #fitness biases not used -- EVO?
                         timer=None):
    """
    The central process that performs a matching or a training round between a generator and a
    discriminator

    :param generator_instance: generator instance used
    :param discriminator_instance: discriminator instance used
    :param disc_optimizer: optimizer used for the discriminator
    :param gen_optimizer: generator used for the generator
    :param criterion: criterion used to calculate divergence between the real and fake samples
    :param dataloader: dataloader feeding the real data in
    :param device: device on which the training is happening  # TRACING: from configs
    :param latent_vector_size: the size of the latent v ector we are using
    :param mode: ``match`` (default) | ``train_g`` (train the generator only) | ``train_d`` (train the
            discriminator only) | ``train`` (train both)
    :param real_label: (optional) = 1  # TRACING: from configs
    :param fake_label: (optional) = 0  # TRACING: from configs
    :param training_epochs: (optional) = 1 for how many epochs to train. if between 0 and 1,
            the batches will be rounded with a ceil to the nearest fraction of the whole dataset.
    :param noise_floor: (optional, inactive) error below which the training is assumed useless
    :param fitness_biases: (optional, inactive) weighting factors for training - ratio of epochs to
            train generator vs discriminatior
    :param timer: (optional) a timer object used to time the execution
    :return: if match: [[average disc error on real, average disc error on fake], ...] for all
            the batches in a single epoch
             if train: [[epoch, batch no, total discrimninator criterion, total discriminator
             criterion on the generator, average success of real dioscriminator on real,
             average error of discriminator on generator before training round, average error of
             discriminator on generator after the training round], ...]
    """

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
            
    #EVO
    if match:
        discriminator_instance.win_rate = 0
        generator_instance.win_rate = 0
                
    
    for epoch in range(training_epochs):

        #EVO
        n = 0
        
        for i, data in enumerate(dataloader, 0):
            
            n += 1
            
            if dataloader_limiter is not None and i > dataloader_limiter:
                break

            if timer is not None:
                timer.start()

                
            # train with real data
            discriminator_instance.zero_grad()
            real_cpu = data[0].to(device)
            _batch_size = real_cpu.size(0)
            label = torch.full((_batch_size,), real_label, device=device, dtype=torch.float32)
            
            #EVO -- perturb the real input of the discriminator
            #perturbation = torch.normal(mean=0, std=0.2, size = real_cpu.size()).to(device)
            #perturbed_input = real_cpu + perturbation.detach()
            
            
            output_on_real = discriminator_instance(real_cpu) #perturbed_input

            errD_real = criterion(output_on_real, label)

            if train_d:
                errD_real.backward()

            average_disc_success_on_real = output_on_real.mean().item()

            # train with fake
            noise = torch.randn(_batch_size, latent_vector_size, 1, 1, device=device)
            fake = generator_instance(noise)  # generates fake data

            label.fill_(fake_label)
                               
            output_on_fake = discriminator_instance(fake.detach())  # flags input as
            # non-gradientable

            errD_fake = criterion(output_on_fake, label)  # calculates the loss for the prediction
            # error

            if train_d:
                errD_fake.backward()  # backpropagates it

            average_disc_error_on_gan = output_on_fake.mean().item()
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

                #EVO
                discriminator_instance.calc_win_rate(output_on_real, output_on_fake)
                
                #generator_instance.calc_win_rate(output_on_fake)   #not needed unless to see how the gen is doing for debug
                
                #EVO -- debug
                #print("BATCH NUMBER: %s, DISCRIMINATOR: %s with win_rate: %s and current_fitness: %s, \
                #GENERATOR: %s with win_rate: %s and current_fitness: %s" \
                #      % (n, discriminator_instance.random_tag, discriminator_instance.win_rate, discriminator_instance.current_fitness,\
                #         generator_instance.random_tag, generator_instance.win_rate, generator_instance.current_fitness))
                
                        
            
        #EVO 
        if match:
            discriminator_instance.win_rate /= n
            generator_instance.win_rate = 1 - discriminator_instance.win_rate
                

    #EVO        
    if match:
        
        discriminator_instance.calc_skill_rating(generator_instance)
        generator_instance.calc_skill_rating(discriminator_instance)
        
    
    
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

    def __init__(self, environment,
                 generator_instance,
                 discriminator_instance,
                 generator_optimizer_partial,
                 discriminator_optimizer_partial,
                 criterion=nn.BCELoss()):
        """
        :param environment: GANEnvironment environment object from the src.match_and_train
        :param generator_instance: the instance of the generator
        :param discriminator_instance: the instance of the discriminator
        :param generator_optimizer_partial: partial function for the optimizer that only requires
                the generator instance parameters and already contains all the hyperparameters
        :param discriminator_optimizer_partial: partial function for the optimizer that only requires
                the discriminator instance parameters and already contains all the hyperparameters
        :param criterion: (optional) loss criterion. by default, BCELoss
        """
        self.env = environment

        self.generator_instance = generator_instance
        self.discriminator_instance = discriminator_instance

        self.generator_optimizer = generator_optimizer_partial(generator_instance.parameters())
        self.discriminator_optimizer = discriminator_optimizer_partial(
            discriminator_instance.parameters())

        self.criterion = criterion

    def match(self, timer=None, commit=True):
        """
        A wrapper for a single match between a discriminator and a generator

        :param timer: (object) timer object
        :param commit: (object) if the training/matching results are to be committed to the database
        :return: output of the ``match_training_round`` function in the match mode
                Aka [[average disc error on real, average disc error on fake], ...] for all
                the batches in a single epoch
        """
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

        #TODO: add the weightings by autoimmunity and virulence
        # print('debug: inside match: real_error: %s, false_error: %s' % (trace[0], trace[1]))

        
        #EVO        
        pathogen_fitness = self.generator_instance.current_fitness
        self.discriminator_instance.real_error = trace[0]
        
        
        
        # print('debug: inside match: host_fitness: %s, pathogen_fitness: %s' % (host_fitness, pathogen_fitness))

        
        # TODO: check for conflict with the decisions made inside the arena module
        if pathogen_fitness > 1100:
            # print('debug: inside match: contamination branch')  # contamination
            '''
            self.generator_instance.fitness_map = {
                self.discriminator_instance.random_tag: pathogen_fitness}
            self.discriminator_instance.gen_error_map = {self.generator_instance.random_tag:
                                                             trace[1]}
            '''
            
            #EVO -- now appending (no more assignment of last)
            self.generator_instance.fitness_map[self.discriminator_instance] = pathogen_fitness #used to be instance.random_tag
            
            self.discriminator_instance.gen_error_map[self.generator_instance] = trace[1] #used to be instance.random_tag
                        
        
        else:  # No contamination
            # print('debug: inside match: no-contamination branch')
            # clear pathogens if exist
            self.generator_instance.fitness_map.pop(self.discriminator_instance, None) #used to be instance.random_tag
            self.discriminator_instance.gen_error_map.pop(self.generator_instance, None) #used to be instance.random_tag
                
        
        if commit:
            self.commit_disc_gen_updates()

        return trace

    def commit_disc_gen_updates(self):
        """
        A helper function that commits the updates to the generator/discriminator performed in
        the database to the mongodb

        :return:
        """
        update_pure_disc(self.discriminator_instance.random_tag,
                         {'encounter_trace': self.discriminator_instance.encounter_trace,
                          'self_error': self.discriminator_instance.real_error,
                          'gen_error_map': self.discriminator_instance.gen_error_map,
                          'current_fitness': self.discriminator_instance.current_fitness})
        update_pure_gen(self.generator_instance.random_tag,
                        {'encounter_trace': self.generator_instance.encounter_trace,
                         'fitness_map': self.generator_instance.fitness_map})

    def first_disc_gen_commit(self, gen_only=False, disc_only=False):
        """
        A helper function that performs a first commit of the generator/discriminator discriminator
        pair into the database

        :param gen_only: if only the generator is to be inserted into the database
        :param disc_only: if only the discriminator is to be inserted into the datasbase
        :return:
        """
        if not disc_only:
            save_pure_gen(self.generator_instance.save_instance_state())
        if not gen_only:
            save_pure_disc(self.discriminator_instance.save_instance_state())
    
    def cross_train(self, epochs=1, gen_only=False, disc_only=False, timer=None, commit=False):
        """
        A wrapper function that allows to train gan and disc one against each other, together or
        separately

        :param epochs: the number of epochs to train for. If decimal, a fraction of the batches
            of the data closest to the decimal will be used, rounding upwards (1 by default)
        :param gen_only: if only generator is to be trained (False by default)
        :param disc_only: if only the disc is to be trained (False by default)
        :param timer: (optional) a timer object
        :param commit: if a commit to the database is to be performed.
        :return: output of the ``match_training_round`` function in the train mode
            Aka [[epoch, batch no, total discrimninator criterion, total discriminator
             criterion on the generator, average success of real dioscriminator on real,
             average error of discriminator on generator before training round, average error of
             discriminator on generator after the training round], ...]
        """

        mode = "train"

        if gen_only and disc_only:
            raise Exception('Both Gan and Disc training are set to only')
        if gen_only:
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

        #EVO -- should we take this before the train? (perturbations before the learning)
        #aneuploidization(self.generator_instance)
        #aneuploidization(self.discriminator_instance)
        #log_normal_aneuploidization(self.generator_instance)
        #log_normal_aneuploidization(self.discriminator_instance)
    
        #EVO -- generation change
        print()
        print('disc: ', self.discriminator_instance.random_tag, '->', end='')
        self.discriminator_instance.bump_random_tag()
        print(self.discriminator_instance.random_tag)
        print('gen: ', self.generator_instance.random_tag, '->', end='')
        self.generator_instance.bump_random_tag()
        print(self.generator_instance.random_tag)
        
        
        if commit:
            self.first_disc_gen_commit(disc_only=disc_only, gen_only=gen_only)

        return trace

    def sample_images(self, annotation=''):
        """
        Save a sample of images that can be generated by the generator at the current stage.

        :param annotation: any annotation to be added to the default folder name where to save
        the images are sampled into
        :return:
        """

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
