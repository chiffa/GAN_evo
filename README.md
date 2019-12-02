# GAN evolutionary arena

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
 
