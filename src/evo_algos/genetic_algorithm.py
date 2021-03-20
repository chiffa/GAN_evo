import numpy as np
import torch.nn as nn

#Part of the project dedicated to genetic/evolutionary algorithms.

def select_mating_pool(arena_fitness_array, selection_option=""):
    # compute the fitness penalty imposed by the model complexity
    # the best so far would be to look at the single forwards pass time
    pass


def crossover(parent_parameter_array):
    pass


def mutate(parameter_array,
           mutation_intensity=1,
           mutation_distribution=lambda x: np.random.gammma(x, 1),
           mutation_policy={'del':1, 'dup':1, 'nl_dup':1.5, 'ins':2, 'met':1, 'sub':0.5}):
    """
    Selects a mutation intensity based on the specified distribution and intensity parameter,
    then randomly selects mutation based on the mutation policy. Finally, deep-copies the supplied
    parameter array and returns a mutated one

    :param parameter_array:
    :param mutation_intensity:
    :param mutation_distribution:
    :param mutation_policy:
    :return:
    """

    def perform_deletion():
        """
        Deletes a layer in the parameter array and corrects for the dimensions.y
        :return:
        """
        pass

    def perform_duplication():
        """
        Duplicates locally (left or right) a randomly selected layers + NL function
        :return:
        """
        pass

    def perform_non_local_duplication():
        """
        Duplicates a layer + NL function, then inserts it randomly
        :return:
        """
        pass

    def perform_insertion():
        """
        Inserts a randomly selected pair of layer (non-linear activation + linear) at a
        random location
        :return:
        """
        pass

    def perform_metaparameter_mutation():
        """
        Mutates the randomly selected meta-parameter
        :return:
        """
        pass

    def perform_subparameter_mutation():
        """
        Mutates the parameters of a randomly selected layer, then propagates it on the
        subsequent/prior layers until a shape modification layer is encountered.
        :return:
        """
        pass



    pass

