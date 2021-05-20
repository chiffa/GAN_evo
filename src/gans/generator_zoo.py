import torch
import torch.nn as nn
from src.gans.nn_structure import NetworkStructure
from random import sample
import string
import pickle
import torchvision.utils as vutils
from src.mongo_interface import pure_gen_from_random_tag
import io
from configs import cuda_device


from src.glicko2 import glicko2



#Generator's implementation

#Generator collection. Use of convolution well adapted to images but not text
#Same storage/ressurection code as for the discriminator

char_set = string.ascii_uppercase + string.digits

# TODO: make sure that the saving and resurrection are done to CPU at first and then sent to CUDAs
torch.cuda.set_device(cuda_device)

def generate_hyperparameter_key(_self):
    key = {'random_tag': _self.random_tag,
           'gen_type': type(_self).__name__,
           'gen_latent_params': _self.generator_latent_maps}
    return key


def save(_self):
    _self.to(torch.device('cpu'))
    key = _self.generate_hyperparameter_key()
    payload = {'encounter_trace': _self.encounter_trace,
               'gen_state': pickle.dumps(_self.state_dict()),
               'fitness_map': _self.fitness_map}

    key.update(payload)
    _self.to(torch.device(cuda_device))

    return key


def resurrect(_self, random_tag):
    _self.to(torch.device('cpu'))

    stored_gen = pure_gen_from_random_tag(random_tag)
    if stored_gen['gen_type'] != type(_self).__name__:
        raise Exception('Wrong class: expected %s, got %s' % (type(_self).__name__,
                                                              stored_gen['gen_type']))
    _self.random_tag = random_tag
    _self.generator_latent_maps = stored_gen['gen_latent_params']
    _self.encounter_trace = stored_gen['encounter_trace']
    # print('encounter_trace:', _self.encounter_trace)
    # fake_file = io.BytesIO(stored_gen)
    # _self.load_state_dict(torch.load(fake_file, map_location=torch.device('cpu')))
    _self.load_state_dict(pickle.loads(stored_gen['gen_state']))
    _self.fitness_map = stored_gen['fitness_map']

    _self.to(torch.device(cuda_device))


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
        
        #EVO
        self.win_rate = 0
        self.glicko = glicko2.Glicko2(mu=1500, phi=350, sigma=0.06, tau=0.3)
        self.skill_rating = self.glicko.create_rating()
        self.skill_rating_games = []
        
        self.current_fitness = self.skill_rating.mu
        
        self.adapt = False
        self.adapted_parent = False
        self.silent_adaptation = False
        self.silent_parent = False
        # ----       
        
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

    def resurrect(self, random_tag):
        return resurrect(self, random_tag)

    #EVO
    def bump_random_tag(self):
        temp_a = self.adapt #save adaptation of parent-to-be
        temp_s = self.silent_parent #save whether the parent-to-be was silently adapted
        
        self.random_tag = ''.join(sample(char_set * 10, 10))
        self.tag_trace += [self.random_tag]
        
        self.adapt = False
        self.adapted_parent = temp_a
        self.silent_adaptation = False
        self.silent_parent = temp_s
        self.fitness_map.clear()
        

    #adds (summation) the average win rate of last batch of images
    def calc_win_rate(self, disc_decision_on_fake):
        self.win_rate += sum((disc_decision_on_fake >= 0.5).float()).item()/len(disc_decision_on_fake)
        
        
    #creates a rating object for its adversarial (its opponent in a specific game)
    #appends (self.win_rate, adv.skill_rating) to its skill_rating_games []
    def calc_skill_rating(self, adversarial):       
        rating = self.glicko.create_rating(mu=adversarial.skill_rating.mu,\
                                           phi=adversarial.skill_rating.phi, sigma=adversarial.skill_rating.sigma)
        
        self.skill_rating_games.append((self.win_rate, rating))

    
    #assigns a skill_rating to self, and resets own skill_rating_games table (got the skill rating from all those stored games played)
    def finish_calc_skill_rating(self):
        self.skill_rating = self.glicko.rate(self.skill_rating, self.skill_rating_games)
        self.skill_rating_games = []
        