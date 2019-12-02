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