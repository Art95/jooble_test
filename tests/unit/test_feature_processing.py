import unittest
import random

from tests.unit.utils import *
from datastats import StatisticsExtractor

class TestFeatureProcessing(unittest.TestCase):
    def setUp(self):
        random.seed(1234)
        self.test_file_path = 'fixtures/test_file.tsv'
        test_data, feature_codes = read_test_data(self.test_file_path)

        self.test_data = test_data
        self.feature_codes = feature_codes

    def test_get_z_score(self):
        self.assertTrue(False)

    def test_get_argmax(self):
        self.assertTrue(False)

    def test_get_abs_max_mean_diff(self):
        self.assertTrue(False)

    def test_process_features_file(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()