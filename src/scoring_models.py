import numpy as np
from scipy import stats


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
                          effective_phenotype_space_dimensions=80):
    """
    The reason that we are using Weibull is due to the existence of the historical reasons of
    the Fisherian phenotypic model.

    We assume that the only adverse effect on the host fitness from the pathogen is due to his
    reproduction, involving host cells break-up.

    the 20 is equivalent to about 5% error starting to be seriously problematic

    """

    cumulative_load = np.sqrt(np.power(real_av_error * autoimmunity_factor, 2)
                                 + np.power(fake_av_error * virulence_factor, 2))

    host_fitness = stats.weibull_min.cdf(cumulative_load, effective_phenotype_space_dimensions)

    pathogen_fitness = virulence_factor * stats.weibull_min.cdf((1-fake_av_error) * virulence_factor * 2,
                                                                effective_phenotype_space_dimensions)

    return host_fitness, pathogen_fitness


def cumulative_host_fitness(real_av_error, fake_av_error_vector,
                        autoimmunity_factor=20, virulence_factor_vector=[],
                        effective_phenotype_space_dimensions=80):

    if len(virulence_factor_vector) == 0:
        virulence_factor_vector = [20 for _ in fake_av_error_vector]

    cumulative_load = np.sum(np.array([(av_err*vir_fac)**2 for av_err, vir_fac in
                                         zip(fake_av_error_vector, virulence_factor_vector)]))

    cumulative_load += np.power(real_av_error * autoimmunity_factor, 2)
    cumulative_load = np.sqrt(cumulative_load)

    host_fitness = stats.weibull_min.cdf(cumulative_load, effective_phenotype_space_dimensions)

    return host_fitness
