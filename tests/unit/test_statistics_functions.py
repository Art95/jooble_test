import unittest
import random

from tests.unit.utils import *
from feature_processor import update_statistics, merge_statistics

class TestStatisticsExtractor(unittest.TestCase):
    def setUp(self):
        random.seed(1234)
        self.test_file_path = 'fixtures/test_file.tsv'
        test_data, feature_codes = read_test_data(self.test_file_path)

        self.test_data = test_data
        self.feature_codes = feature_codes

    def test_update_statistics(self):
        feature_codes = random.sample(self.feature_codes, k=2)
        filtered_rows = filter_features_for_codes(self.test_data, feature_codes)
        stats = { "mean": np.array([]),
                  "std": np.array([]),
                  "min": np.int32,
                  "max": np.int32,
                  "count": np.int32 }

        for i, features in enumerate(filtered_rows):
            counter = i + 1
            current_rows = filtered_rows[:counter][:]

            expected_result = { "count": counter,
                                "mean": current_rows.mean(axis=0),
                                "std": current_rows.std(axis=0, ddof=1), # sample standard deviation,
                                "max": np.amax(current_rows, axis=0),
                                "min": np.amin(current_rows, axis=0) }

            actual_result = update_statistics(features, stats)

            self.assertEqual(expected_result, actual_result)


    def test_merge_statistics(self):
        feature_codes = random.sample(self.feature_codes, k=3)
        stats_for_merging = []

        for feature_code in feature_codes:
            features_for_code = filter_features_for_code(self.test_data, feature_code)


            code_stats = { "count": features_for_code.shape[0],
                           "mean": features_for_code.mean(axis=0),
                           "std": features_for_code.std(axis=0, ddof=1), # sample standard deviation
                           "min": np.amin(features_for_code, axis=0),
                           "max": np.amax(features_for_code, axis=0) }

            stats_for_merging.append(code_stats)

        combined_rows = filter_features_for_codes(self.test_data, feature_codes)

        expected_result = { "count": combined_rows.shape[0],
                            "mean": combined_rows.mean(axis=0),
                            "std": combined_rows.std(axis=0, ddof=1), # sample standard deviation
                            "max": np.amax(combined_rows, axis=0),
                            "min": np.amin(combined_rows, axis=0) }

        actual_result = merge_statistics(stats_for_merging)

        self.assertEqual(expected_result, actual_result)


if __name__ == '__main__':
    unittest.main()