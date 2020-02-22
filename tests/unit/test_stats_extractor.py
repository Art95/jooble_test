import unittest
import random
import numpy as np
from datastats import StatisticsExtractor

class TestStatisticsExtractor(unittest.TestCase):
    def setUp(self):
        random.seed(1234)
        self.test_file_path = 'fixtures/test_file.tsv'
        self.test_data = {}
        unique_feature_codes = set()

        # Load test data
        with open(self.test_file_path) as data_file:
            for line in data_file:
                columns = line.split('\t')
                values = columns[1].split(',')

                job_id = np.int64(columns[0])
                feature_code = np.int32(values[0])
                features = np.array(values[1:], dtype=np.int32)

                if job_id not in self.test_data:
                    self.test_data[job_id] = {}

                self.test_data[job_id][feature_code] = features
                unique_feature_codes.add(feature_code)

        self.feature_codes = list(unique_feature_codes)

    def test_get_statistics_one_feature_code(self):
        feature_code = 2
        stats_extractor = StatisticsExtractor(self.test_file_path)
        filtered_rows = self._filter_features_for_code(feature_code)

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
        filtered_rows = self._filter_features_for_codes(self.feature_codes)

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
        filtered_rows = self._filter_features_for_codes(feature_codes)

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

    def _filter_features_for_codes(self, feature_codes):
        filtered_features = np.array([])

        for feature_code in feature_codes:
            features_for_code = self._filter_features_for_code(feature_code)
            filtered_features = np.vstack([filtered_features, features_for_code]) \
                if filtered_features.size else features_for_code

        return filtered_features

    def _filter_features_for_code(self, feature_code):
        filtered_features = np.array([])

        for job_id in self.test_data:
            if feature_code in self.test_data[job_id]:
                features = self.test_data[job_id][feature_code]
                filtered_features = np.vstack([filtered_features, features]) \
                    if filtered_features.size else features

        return filtered_features

if __name__ == '__main__':
    unittest.main()