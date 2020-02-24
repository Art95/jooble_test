import unittest
import random
from pathlib import Path
import os

from tests.unit.utils import *
from scipy import stats

import feature_processor

class TestFeatureFunctions(unittest.TestCase):
    def setUp(self):
        random.seed(1234)
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))

        self.test_file_path = current_path / 'fixtures'/ 'test_input.tsv'
        test_data, feature_codes = read_test_data(self.test_file_path)

        self.test_data = test_data
        self.feature_codes = feature_codes

    def test_calculate_z_score(self):
        features = filter_features_for_codes(self.test_data, self.feature_codes)
        means = features.mean(axis=0)
        stds = features.std(axis=0, ddof=1)

        expected_z_scores = stats.zscore(features, axis=0, ddof=1) # using sample standard deviation

        for i, feature_row in enumerate(features):
            actual_z_cores = feature_processor.calculate_z_score(feature_row, means, stds)
            np.testing.assert_allclose(actual_z_cores, expected_z_scores[i])


    def test_get_argmax(self):
        features = filter_features_for_codes(self.test_data, self.feature_codes)

        for feature_row in features:
            expected_argmax = feature_row.argmax()
            actual_argmax = feature_processor.get_argmax(feature_row)

            self.assertEqual(expected_argmax, actual_argmax)

    def test_calculate_abs_max_mean_diff(self):
        features = filter_features_for_codes(self.test_data, self.feature_codes)
        means = np.mean(features, axis=0)

        for feature_row in features:
            index_max = feature_row.argmax()
            expected_abs_max_mean_diff = np.abs(feature_row[index_max] - means[index_max])
            actual_abs_max_mean_diff = feature_processor.calculate_abs_max_mean_diff(feature_row, means)

            self.assertEqual(expected_abs_max_mean_diff, actual_abs_max_mean_diff)

    def test_get_min(self):
        features = filter_features_for_codes(self.test_data, self.feature_codes)

        for feature_row in features:
            expected_min = np.amin(feature_row)
            actual_min = feature_processor.get_min(feature_row)

            self.assertEqual(expected_min, actual_min)

    def test_get_max(self):
        features = filter_features_for_codes(self.test_data, self.feature_codes)

        for feature_row in features:
            expected_max = np.amax(feature_row)
            actual_max = feature_processor.get_max(feature_row)

            self.assertEqual(expected_max, actual_max)

    def test_transform_features(self):
        features = filter_features_for_codes(self.test_data, self.feature_codes)
        means = features.mean(axis=0)
        stds = features.std(axis=0, ddof=1)

        features_stats = {
            "mean": means,
            "std": stds
        }

        expected_z_scores = stats.zscore(features, axis=0, ddof=1)  # using sample standard deviation
        expected_argmaxs = np.argmax(features, axis=1)

        for i, feature in enumerate(features):
            expected_result = {
                "z_scores": expected_z_scores[i],
                "argmax": expected_argmaxs[i],
                "abs_max_mean_diff": np.abs(feature[expected_argmaxs[i]] - means[expected_argmaxs[i]])
            }

            actual_result = feature_processor.transform_features(feature, features_stats)

            for metric in expected_result:
                np.testing.assert_allclose(actual_result[metric], expected_result[metric])

    def test_transform_features_some_codes(self):
        selected_codes = random.sample(self.feature_codes, k=2)

        features = filter_features_for_codes(self.test_data, selected_codes)
        means = features.mean(axis=0)
        stds = features.std(axis=0, ddof=1)

        features_stats = {
            "mean": means,
            "std": stds
        }

        expected_z_scores = stats.zscore(features, axis=0, ddof=1)  # using sample standard deviation
        expected_argmaxs = np.argmax(features, axis=1)

        for i, feature in enumerate(features):
            expected_result = {
                "z_scores": expected_z_scores[i],
                "argmax": expected_argmaxs[i],
                "abs_max_mean_diff": np.abs(feature[expected_argmaxs[i]] - means[expected_argmaxs[i]])
            }

            actual_result = feature_processor.transform_features(feature, features_stats)

            for metric in expected_result:
                np.testing.assert_allclose(actual_result[metric], expected_result[metric])


if __name__ == '__main__':
    unittest.main()