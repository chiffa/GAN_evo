from src.mongo_interface import gan_pair_list_by_filter, gan_pair_update_in_db
from src.gans.trainer_zoo import GanTrainer
import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import combinations


# TODO: two-updates (Generator, then Generator tries to outrun the DISC)
# TODO: Generator pulls in the disc population, tries to bind to some, then gets chased. Discs
#  get cleared if its fitness is too low.


def run_match(dataset, limiter={}):
    participants = []

    for trainer in gan_pair_list_by_filter(limiter):
        participants.append(GanTrainer(dataset, from_dict=trainer))

    for A, B in combinations(participants, 2):
        print("Match between %s and %s"
              "\tstarting disc/gan elo scores: A:%.2f/%.2f\t B:%.2f/%.2f" %
              (A.random_tag, B.random_tag,
               A.disc_elo, A.gen_elo,
               B.disc_elo, B.gen_elo))
        A.match(B)

    print("rankings by discriminator_instance score:")
    disc_score_accumulator = 0
    for trainer in sorted(participants, key=lambda x: x.disc_elo, reverse=True):
        print("\t %s: %.2f" % (trainer.random_tag, trainer.disc_elo))
        disc_score_accumulator += trainer.disc_elo

    print("total discriminator_instance elo score: %.2f" % disc_score_accumulator)

    print("rankings by generator_instance score:")
    gen_score_accumulator = 0
    for trainer in sorted(participants, key=lambda x: x.gen_elo, reverse=True):
        print("\t %s: %.2f" % (trainer.random_tag, trainer.gen_elo))
        gen_score_accumulator += trainer.gen_elo

    print("total generator_instance elo score: %.2f" % gen_score_accumulator)

    print("models meta-parameters:")
    for trainer in participants:
        print('training %s with following parameter array: '
              'bs: %s, dlv: %s, glv: %s, '
              'lr: %.5f, b: %.2f, tep: %s' % (trainer.random_tag,
                                              trainer.batch_size, trainer.latent_vector_size,
                                              trainer.generator_latent_maps,
                                              trainer.learning_rate, trainer.beta1,
                                              trainer.training_epochs))


def reset_scores(dataset, limiter={}):
    participants = []

    for trainer in gan_pair_list_by_filter(limiter):
        gan_pair_update_in_db({'random_tag': trainer['random_tag']},
                              {'score_ratings': (0, 1500, 1500)})


def sample_generator_images(dataset, limiter={}):
    for trainer in gan_pair_list_by_filter(limiter):
        trainer_instance = GanTrainer(dataset, from_dict=trainer)
        trainer_instance.sample_images('gen . disc Elo: %.2f . %2.f' % (trainer_instance.gen_elo,
                                                                     trainer_instance.disc_elo))


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

    # reset_scores(dataset=mnist_dataset)
    # run_match(dataset=mnist_dataset)
    sample_generator_images(mnist_dataset)
