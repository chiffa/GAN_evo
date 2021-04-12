# GAN evolutionary arena

## General structure: 

### `Arena` object
The raw generator/discriminator structures are stored in the `src.gans.
<generator/discriminator>_zoo`. Several convenience functions, such self-saving and 
self-resurrection to and from text-serialized forms are implemented as monkey-patching in there 
as well.

However, the training requires more paramters, such as data, hyper-parameters, optimizers, 
criterion, .... All of that is stored in the `src.gans.match_and_train` `GANEnvironment` object.

The actual evaluation of performance of a generator against a discriminator as well as the 
logistics of training between them are managed by the `match_training_round` function in 
the `src.gans.match_and_train` module, and are in turn wrapped into the `Arena.match` and 
`Arena.cross_train` methods of the `Arena` object, that integrates the environment, generator 
and discriminator instances as well as the loss criterions, and the optimizers used by both. 
Basically, `Arena` is the interface that is used for any training/adversarial performance 
evaluation of the generator/discriminator instances. For convenience, `Arena` is also the object 
that contains the method `Arena.sample_images`, allowing the loaded generator to generate sample 
images.


### Memorization of the run.
 - Runs are saved into a mongoDB instance (`src.mongo_interface`).
 - The saving per se is performed by the `save_pure_<gen/disc>` function
    - its call is wrapped by the `save_pure_<gen/disc>_helper` function, that:
        - dumps the gen/disc instance state with a (`<gen/disc>.save_instance_state()`)
            - `<gen/disc>.save_instance_state()` is a monkeypatch for a `save(_self)` function
            - `save(_self)`
                - gets a hyper_parameters key (`self.random_tag`, `gen_type`, `gen_latent_params`)
                - adds to it the encounter trace, generator state and the fitness map
                  (`self.encounter_trace`, `self.state_dict()`, `self.fitness_map`)
                - and transfers it to the `cpu` torch device for the portability
            - `resurrect(_self)` is the pendant of the `save` and is mokeypatched directly as 
              `<gen/disc>.resurrect()`
              
    - gen/disc per se are saved into the `pure_disc`/`pure_gen` collections
    - the encounter trace is split off and is saved into its separate collection 
      (`pure_match_trace`). The MongoDB Id of the trace insertion object is then saved into the 
      `pure_<disc/gen>` onjects
 - the trace of the encounters, is ho


## Prior work:
Evolutionary search on the networks is a fairly well explored idea. However, using the concepts
 from population/ecological evolution is significantly less common. 
 
Here, our goal is to look whether we can detect collaborative/antagonistic evolution patterns in
 GANs. 

We would like to explore the following modalities of evolution:
 - Lying discriminators (Byzantine discriminators)
 - Role of sexes - Papadimitriou's mixability
 - Speciation (players having a pool of GANs and discriminators that report other's GANs, but not
  their own)
 - Small mutations vs large (aneuploidization)
 - Tolerance to low performers (importance of variability)
 - Spread - bottleneck - spread - bottleneck (soft swipes)
 - RAC curves weighting effect - letting some fakes through vs blocking the true ones.
 - algorithms for the evolution: Genetic vs Particle Swarm optimization
 - Optimization to induce into error a single discriminator rather than a cluster of them.
 - importance of the standing variation vs the one emerging under immediate selection

## Parameters so far: 
 - Structure of the layers (In>NLF>Conv>NLF>Full_conn>NLF>Out)
 - Parameters of the layers 
 - Inheritance of weights from previous iterations? Training on the GAN/Disc pair owned by the
  players?
  
## Evolution meta-parameters:
  - probability of mutation
  - mutation magnitude prob
    - How do we determine that magnitude? Idea - micro-parameters change magnitude.
  - 
 
 
## Life deploy testing:
 - detection of adverse comments on eg. Discord servers (cf MalwareTech).
 
## Potential future expansion:
 - co-linearity of the viral infection estimated by the random keys from which they were generated
 (Host-pathogen gans angle)
 
 
## Evolutionary adaptative bursts:
 - large changes in parameters upon environment change?
