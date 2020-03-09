from src.gans.discriminator_zoo import Discriminator, Discriminator_PReLU, Discriminator_with_full_linear
from src.gans.generator_zoo import Generator
from src.gans.trainer_zoo import Arena, GANEnvironment
# from src.new_mongo_interface import save_pure_disc, save_pure_gen, filter_pure_disc, filter_pure_gen

import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import combinations
import torch.optim as optim

if __name__ == "__main__":
    image_folder = "./image"
    image_size = 64
    number_of_colors = 1
    imtype = 'mnist'

    mnist_dataset = dset.MNIST(root=image_folder, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),]))

    environment = GANEnvironment(mnist_dataset)

    gen = Generator(ngpu=environment.ngpu,
                    latent_vector_size=environment.latent_vector_size,
                    generator_latent_maps=64,
                    number_of_colors=environment.number_of_colors).to(environment.device)

    disc = Discriminator(ngpu=environment.ngpu,
                         latent_vector_size=environment.latent_vector_size,
                         discriminator_latent_maps=64,
                         number_of_colors=environment.number_of_colors).to(environment.device)

    learning_rate = 0.0002
    beta1 = 0.5

    gen_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))
    disc_opt_part = lambda x: optim.Adam(x, lr=learning_rate, betas=(beta1, 0.999))

    arena = Arena(environment=environment,
                  generator_instance=gen,
                  discriminator_instance=disc,
                  generator_optimizer_partial=gen_opt_part,
                  discriminator_optimizer_partial=disc_opt_part,
                  )

    arena.cross_train(10)
    arena.sample_images()
    print(arena.generator_instance.random_tag)
    print(arena.match())
