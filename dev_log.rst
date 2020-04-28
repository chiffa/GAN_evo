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
    => Right now we will be just doing the linear map adapter to resize it all
TODO: the final/starting layers seem to be very different in GANs and discriminators.
    => it's mostly shapes.

We also have meta-parameters that we can/should be optimizing:
    - optimization algorithm
    - learning rate

And a sub-parameter; which is the base variation vector. although, is it involved in the
classification?

Can we talk about the beneficial vs nefarious mutation distribution?

TODO: the problem is that the convolutional levels perform a modificaiton of the size that is not
 trivial. It's basically dims_out = (dims_in + padding * 2 - kernel_size + stride) // stride

Solution: perform a resize from the tensorflow toolkit.

After some testing:
===================
- GANs seem to overfit by "hacking" the reward of its discriminator.
- Which means that we are getting a competition from different architecture that would not overfit.
- Once we did one-to-one training, we perform all GANs vs all discriminators round, training them
 once and then selecting Discriminators on the second round.

For the efficiency, the single pairs need to be trained separately, and then their models saved
and then send into the arena on their own.

Application logistics:
=======================
DONE - Logistics to store the gans and discriminators (mongodb)
DONE - for that we can use the state dict of pytorch and move it in and out of mongodb,
pulling in
     and out of the python instances with a model.state_dict() and model.load_state_dict()
     model.eval()
? - Logistics to pipe some of the inputs into some of the outputs
DONE - The training pair should be done in the module, binding temporary parameters overall.
DONE - In the current configuration, we can start parallel training of the pairs that are mapped
with signature of training parameters + unique ID, then stored in a mongodb

- We now need to spin up a method to find all the image-type associated GAN pairs, filter by
fitness and perform a match round between them.
DONE: - Add a numpy array to store various metrics, add it to the GAN pair graph and store in DB.
EH Not needed now: -Plot it as well.

- TODO: add saving of the discriminator/generator + training traces to the disk. Mongod should only
contain the pointer to the path (that breaks containerization though)

- TODO: add support for cross-training the discriminator/generator

- TODO: add support for the multi-generator pull in the individual trainer (lists)

- Done: move the training, matching and cross-training out of self into the arena level. replace
self by an (generator_supplier_instance, descriminator_supplier_instance, gen_optimizer,
disc_optimizer, criterion) => training trace + internal object modification/saving

That would allow a single function, unifying the match, training and cross-training

- TODO: Add a random noise layer properly


- Refactoring is proving to be a bit more challenging. Saving is direct, but with multi-type dicts,
and the environment factored out, we need a high-level mixer to pull them all together into
recoverable elements at later stages.

- Similarly, storage/de-storage requires an injection from environment - so within an arena

- Similarly, traces now require enforced ordering, unless we start forking off aggressively. Which
might be a plan actually.

Containerization:
=================
We will need to manage a cluster of containers on the proper cloud with the help of python script if
 we are to deploy. right now we synchronize on the genetic_algo. Ideally, once finished training,
 every new algo will pull all available opponnents and decide from whom he will be inheriting next.
 => Asynchronius fight


Critical modifications to the architecture:
===========================================
- Restarts of training - on-the-local filesystem storage (minimize the latency)
- Commit to DB only the last generation pair
- Generate separate run dump csv files, then stitch them before analysis.

Pulled from the LaTeX:
======================
- Coninfection?
- Vaccination/re-infection?
- Two-phase training - generator starts, gets an epoch and then gets chased by the discriminator.