from src.mongo_interface import gan_pair_list_by_filter
from src.gans.gan_disc import GanTrainer
import torchvision.datasets as dset
import torchvision.transforms as transforms
from itertools import combinations


def run_match(dataset, limiter={}):
    participants = []

    for trainer in gan_pair_list_by_filter(limiter):
        participants.append(GanTrainer(dataset, from_dict=trainer))

    for A, B in combinations(participants, 2):
        print("Match between %s and %s"
              "\starting elo scores: A:%.2f\t B:%.2f" % (A.random_tag, B.random_tag,
                                                         A.elo, B.elo))
        A.match(B)

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
    run_match(dataset=mnist_dataset)
