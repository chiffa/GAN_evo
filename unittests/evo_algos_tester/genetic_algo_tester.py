import unittest
from src.evo_algos import genetic_algorithm as ga


class Genetic_Algo_Tester(unittest.TestCase):

    def test_partner_selection(self):
        self.assertEqual(True, False)


class MutationValidityTester(unittest.TestCase):

    def test_layer_mutation_validity(self):
        self.assertEqual(True, False)

    def test_network_stucture_mutation_validity(self):
        self.assertEqual(True, False)

if __name__ == '__main__':
    unittest.main()
