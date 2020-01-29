import numpy as np


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


def virus_host_fitness(self_errD_real, oponnent_errD_real,
                       self_gen_opp_disc_av_err, self_gen_self_disc_av_err,
                       opp_gen_self_disc_av_err, opp_gen_opp_disc_av_err,
                       autoimmunity_factor_A, autoimmunity_factor_B,
                       virulence_factor_A, virulence_factor_B):
    pass


def elo_algs_wrapper(self_errD_real, self_gen_self_disc_errD, opp_gen_self_disc_errD,
                     oponnent_errD_real, opp_gen_opp_disc_errD, self_gen_opp_disc_errD,
                     self_gen_opp_disc_av_err, self_gen_self_disc_av_err,
                     opp_gen_self_disc_av_err, opp_gen_opp_disc_av_err
                     ):
    pass
