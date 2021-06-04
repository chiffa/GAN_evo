import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


#Functions returning the host and pathogen's fitnesses


host_low_fitness_clip = 0.01
pathogen_low_fitness_clip = 0.05


def simple_elo(elo_A, elo_B, A_lead, factor_k=16):
    disc_e_a = 1. / (1. + 10 ** ((elo_B - elo_A) / 400))
    disc_e_b = 1. / (1. + 10 ** ((elo_A - elo_B) / 400))

    new_elo_A = elo_A + factor_k * (A_lead - disc_e_a)
    new_elo_B = elo_B + factor_k * (- A_lead - disc_e_b)

    return new_elo_A, new_elo_B


def weighted_elo(elo_A, elo_B, A_lead, factor_k_function=lambda x: 16, ):
    disc_e_a = 1. / (1. + 10 ** ((elo_B - elo_A) / 400))
    disc_e_b = 1. / (1. + 10 ** ((elo_A - elo_B) / 400))

    new_elo_A = elo_A + factor_k_function(A_lead) * (A_lead - disc_e_a)
    new_elo_B = elo_B + factor_k_function(A_lead) * (-A_lead - disc_e_b)

    return new_elo_A, new_elo_B


def log_weighted_elo(elo_A, elo_B, A_lead, base_k=16):

    multiplier_function = lambda x: base_k * np.log(abs(x))

    return weighted_elo(elo_A, elo_B, A_lead,
                        factor_k_function=multiplier_function)


def pathogen_host_fitness(real_av_error, fake_av_error,
                          autoimmunity_factor=20, virulence_factor=20,
                          effective_phenotype_space_dimensions=2):
    """
    The reason that we are using Weibull is due to the existence of the historical reasons of
    the Fisherian phenotypic model.

    We assume that the only adverse effect on the host fitness from the pathogen is due to his
    reproduction, involving host cells break-up.

    the 20 is equivalent to about 5% error starting to be seriously problematic

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


#TODO: alternative fitness (proportional to capacity of error induction)
def cumulative_host_fitness(real_av_error, fake_av_error_vector,
                        autoimmunity_factor=20, virulence_factor_vector=[],
                        effective_phenotype_space_dimensions=2):

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