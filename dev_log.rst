Project structure:
==================

We have four main components - GAN/Disc pair; evolutionary algorithm; arena and players.

GAN(s) is/are trained to best fit (a) discriminator(s) that the player has control on.

In the arena, the players pit GANs against the discriminators of other players.

GAN's fitness is proportional to their ability to confuse opposing discriminators. If all other
player's discriminators determined that a given GAN is fake, it's fitness is 0. If no other's
discriminators pick up their GAN, its fitness is maximal.

The elimination curve is the spread that shows how prevalent/taken by other players the GAN is
once the arena game finished (eg we had a human/expert step in, properly label the images and
send them back to the training set).

Crossover - basically appears in the segments of the neural networks where before that point NN1
is taken and after NN2 is taken. Then mutation is applied, be it at a large scale or at a small
scale.
    => Start with uniform distribution.
    => mutation can either be addition/deletion of a layer
        - which is preferably a duplication of an existing layer, if addition.
    => Or modification of a layer parameter.

TODO: how do we modify the width of a layer? GANs seem to be bottle-necked.

Can we talk about the beneficial vs nefarious mutation distribution?