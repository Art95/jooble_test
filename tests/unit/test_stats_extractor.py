import unittest
import random

from tests.unit.utils import *
from datastats import StatisticsExtractor

class TestStatisticsExtractor(unittest.TestCase):
    def setUp(self):
        random.seed(1234)
        self.test_file_path = 'fixtures/test_file.tsv'
        test_data, feature_codes = read_test_data(self.test_file_path)

        self.test_data = test_data
        self.feature_codes = feature_codes

    def test_get_statistics_one_feature_code(self):
        feature_code = 2
        stats_extractor = StatisticsExtractor(self.test_file_path)
        filtered_rows = filter_features_for_code(self.test_data, feature_code)

        expected_means = filtered_rows.mean(axis=0)
        expected_stds = filtered_rows.std(axis=0, ddof=1) # sample standard deviation
        expected_maxs = np.amax(filtered_rows, axis=0)
        expected_mins = np.amin(filtered_rows, axis=0)
        expected_counter = filtered_rows.shape[0]

        expected_result = { "count": expected_counter,
                            "mean": expected_means,
                            "std": expected_stds,
                            "max": expected_maxs,
                            "min": expected_mins }

        actual_result = stats_extractor.get_statistics(codes=[feature_code])

        self.assertEqual(expected_result, actual_result)


    def test_get_statistics_all_feature_codes(self):
        stats_extractor = StatisticsExtractor(self.test_file_path)
        filtered_rows = filter_features_for_codes(self.test_data, self.feature_codes)

        expected_means = filtered_rows.mean(axis=0)
        expected_stds = filtered_rows.std(axis=0, ddof=1) # sample standard deviation
        expected_maxs = np.amax(filtered_rows, axis=0)
        expected_mins = np.amin(filtered_rows, axis=0)
        expected_counter = filtered_rows.shape[0]

        expected_result = { "count": expected_counter,
                            "mean": expected_means,
                            "std": expected_stds,
                            "max": expected_maxs,
                            "min": expected_mins }

        actual_result = stats_extractor.get_statistics(codes=self.feature_codes)

        self.assertEqual(expected_result, actual_result)


    def test_get_statistics_some_feature_codes(self):
        feature_codes = random.sample(self.feature_codes, k=3)
        stats_extractor = StatisticsExtractor(self.test_file_path)
        filtered_rows = filter_features_for_codes(self.test_data, feature_codes)

        expected_means = filtered_rows.mean(axis=0)
        expected_stds = filtered_rows.std(axis=0, ddof=1) # sample standard deviation
        expected_maxs = np.amax(filtered_rows, axis=0)
        expected_mins = np.amin(filtered_rows, axis=0)
        expected_counter = filtered_rows.shape[0]

        expected_result = { "count": expected_counter,
                            "mean": expected_means,
                            "std": expected_stds,
                            "max": expected_maxs,
                            "min": expected_mins }

        actual_result = stats_extractor.get_statistics(codes=[feature_codes])

        self.assertEqual(expected_result, actual_result)


if __name__ == '__main__':
    unittest.main()