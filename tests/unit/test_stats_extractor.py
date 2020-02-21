import unittest
import numpy as np
import datastats

class TestStatisticsExtractor(unittest.TestCase):
    def setUp(self):
        self.test_data = {}
        self.feature_codes = set()

        # Load test data
        with open('fixtures/test_file.tsv') as data_file:
            for line in data_file:
                columns = line.split('\t')
                values = columns[1].split(',')

                job_id = np.int64(columns[0])
                feature_code = np.int32(values[0])
                features = np.array(values[1:], dtype=np.int32)

                if job_id not in self.test_data:
                    self.test_data[job_id] = {}

                self.test_data[job_id][feature_code] = features
                self.feature_codes.add(feature_code)

    def test_get_statistics_one_feature_code(self):
        self.assertTrue(False)

    def test_get_statistics_all_feature_codes(self):
        self.assertTrue(False)

    def test_get_statistics_some_feature_codes(self):
        self.assertTrue(False)

    def _filter_features_for_codes(self, feature_codes):
        filtered_features = np.array([])

        for feature_code in feature_codes:
            features_for_code = self._filter_features_for_code(feature_code)
            filtered_features = np.vstack([filtered_features, features_for_code]) \
                if feature_code.size else features_for_code

        return filtered_features

    def _filter_features_for_code(self, feature_code):
        filtered_features = np.array([])

        for job_id in self.test_data:
            if feature_code in self.test_data[job_id]:
                features = self.test_data[job_id][feature_code]
                filtered_features = np.vstack([filtered_features, features]) if feature_code.size else features

        return filtered_features

if __name__ == '__main__':
    unittest.main()