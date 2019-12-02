

class Player(object):

    def __init__(self, gan_pool, dis_pool):
        self.gan_pool = gan_pool
        self.dis_pool = dis_pool

    def train_gans(self):
        pass

    def process_real_data(self, gan_and_real_data):
        pass
        # return gan_prediction_vector, decision_gan_or_real

