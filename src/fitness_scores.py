import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


host_low_fitness_clip = 0.01
pathogen_low_fitness_clip = 0.05


def pathogen_host_fitness(real_av_error, fake_av_error,
                          autoimmunity_factor=20, virulence_factor=20,
                          effective_phenotype_space_dimensions=2):
    """
    The function used to compute the host/pathogen fitnesses from discriminator errors

    :param real_av_error: error of the discriminator on the real images
    :param fake_av_error: error of the discriminator on the generator images
    :param autoimmunity_factor: autoimmunity factor
    :param virulence_factor: viral reproduction number
    :param effective_phenotype_space_dimensions: effective pheontype dimensions
    :return: host fitness (as if it was infected only by that pathogen), pathogen fitness
    """

    cumulative_load = np.sqrt(np.power(real_av_error * autoimmunity_factor, 2)
                                 + np.power(fake_av_error * virulence_factor, 2))

    host_fitness = 1 - stats.weibull_min.cdf(cumulative_load, effective_phenotype_space_dimensions)

    pathogen_fitness = virulence_factor * stats.weibull_min.cdf(fake_av_error * virulence_factor,
                                                                    effective_phenotype_space_dimensions)

    host_fitness_clipped = False
    host_pre_clip_fitness = host_fitness
    if host_fitness < host_low_fitness_clip:
        host_fitness_clipped = True
        host_fitness = host_low_fitness_clip

    pathogen_fitness_clipped = False
    pathogen_pre_clip_fitness = pathogen_fitness
    if pathogen_fitness < pathogen_low_fitness_clip:
        pathogen_fitness_clipped = True
        pathogen_fitness = pathogen_low_fitness_clip

    print('debug: raw scoring called with errors: real:%s, fake:%s; '
          '\n\t cumulative load: %s;'
          '\n\t host/pathogen pre-clip fitnesses: %s, %s'
          '\n\t host/pathogen fitnesses clipped: %s, %s'
          '\n\t fitness returned: host:%s, pathogen:%s' % (
        real_av_error, fake_av_error,
        cumulative_load,
        host_pre_clip_fitness, pathogen_pre_clip_fitness,
        host_fitness_clipped, pathogen_fitness_clipped,
        host_fitness, pathogen_fitness))

    return host_fitness, pathogen_fitness


def cumulative_host_fitness(real_av_error, fake_av_error_vector,
                        autoimmunity_factor=20, virulence_factor_vector=[],
                        effective_phenotype_space_dimensions=2):
    """
    Function to compute the fitness of the host from based on its error on the real images and
    all the pathogens it encountered and was infected by.

    :param real_av_error: error of the discriminator on the real images
    :param fake_av_error: error of the discriminator on the generator images
    :param autoimmunity_factor: autoimmunity factor
    :param virulence_factor: viral reproduction number
    :param effective_phenotype_space_dimensions: effective pheontype dimensions
    :return: host fitness
    """

    if len(virulence_factor_vector) == 0:
        virulence_factor_vector = [20 for _ in fake_av_error_vector]

    cumulative_load = np.sum(np.array([(av_err*vir_fac)**2 for av_err, vir_fac in
                                         zip(fake_av_error_vector, virulence_factor_vector)]))

    cumulative_load += np.power(real_av_error * autoimmunity_factor, 2)
    cumulative_load = np.sqrt(cumulative_load)

    host_fitness = 1 - stats.weibull_min.cdf(cumulative_load, effective_phenotype_space_dimensions)

    fitness_clipped = False
    pre_clip_fitness = host_fitness
    if host_fitness < host_low_fitness_clip:
        fitness_clipped = True
        host_fitness = host_low_fitness_clip

    print('debug: vector scoring called with errors: real:%s, fakes vector:%s;'
          '\n\tcumulative deviation: %s;'
          '\n\traw fitness: %s;'
          '\n\tfitness clipped: %s'
          '\n\tfitness returned: host:%s' % (real_av_error,
                                          fake_av_error_vector,
                                          cumulative_load,
                                          pre_clip_fitness,
                                          fitness_clipped,
                                          host_fitness))

    return host_fitness


if __name__ == "__main__":

    c = 40

    x = np.linspace(stats.weibull_min.ppf(0.0001, c),
                     stats.weibull_min.ppf(0.9999, c), 100)

    plt.plot(x, stats.weibull_min.cdf(x, c))

    plt.show()