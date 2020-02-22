import unittest
import random

from tests.unit.utils import *
from scipy import stats

from datastats import calculate_abs_max_mean_diff, get_argmax, calculate_z_score, get_max, get_min

class TestFeatureProcessing(unittest.TestCase):
    def setUp(self):
        random.seed(1234)
        self.test_file_path = 'fixtures/test_file.tsv'
        test_data, feature_codes = read_test_data(self.test_file_path)

        self.test_data = test_data
        self.feature_codes = feature_codes

    def test_get_z_score(self):
        features = filter_features_for_codes(self.test_data, self.feature_codes)
        means = features.mean(axis=0)
        stds = features.std(axis=0, ddof=1)

        expected_z_scores = stats.zscore(features, axis=0, ddof=1) # using sample standard deviation

        for i, feature_row in enumerate(features):
            actual_z_cores = calculate_z_score(feature_row, means, stds)
            self.assertEqual(expected_z_scores[i], actual_z_cores)


    def test_get_argmax(self):
        features = filter_features_for_codes(self.test_data, self.feature_codes)

        for feature_row in features:
            expected_argmax = feature_row.argmax()
            actual_argmax = get_argmax(feature_row)

            self.assertEqual(expected_argmax, actual_argmax)

    def test_get_abs_max_mean_diff(self):
        features = filter_features_for_codes(self.test_data, self.feature_codes)
        means = np.mean(features, axis=0)

        for feature_row in features:
            index_max = feature_row.argmax()
            expected_abs_max_mean_diff = np.abs(feature_row[index_max] - means[index_max])
            actual_abs_max_mean_diff = calculate_abs_max_mean_diff(feature_row, means)

            self.assertEqual(expected_abs_max_mean_diff, actual_abs_max_mean_diff)

    def test_get_min(self):
        features = filter_features_for_codes(self.test_data, self.feature_codes)

        for feature_row in features:
            expected_min = np.amin(feature_row)
            actual_min = get_min(feature_row)

            self.assertEqual(expected_min, actual_min)

    def test_get_max(self):
        features = filter_features_for_codes(self.test_data, self.feature_codes)

        for feature_row in features:
            expected_max = np.amax(feature_row)
            actual_max = get_max(feature_row)

            self.assertEqual(expected_max, actual_max)

if __name__ == '__main__':
    unittest.main()