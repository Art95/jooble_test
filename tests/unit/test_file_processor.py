import unittest
from pathlib import Path
import os
from scipy import stats
import random

from file_processor import FileProcessor
from tests.unit.utils import *


class TestFileProcessor(unittest.TestCase):
    def setUp(self):
        random.seed(1234)
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))

        self.input_file_path = current_path / 'fixtures' / 'test_input.tsv'
        self.output_file_path = current_path / 'fixtures' / 'tests_output.tsv'
        test_data, feature_codes = read_test_data(self.input_file_path)

        self.test_data = test_data
        self.feature_codes = feature_codes

    def test_transform_features_all_codes(self):
        file_processor = FileProcessor(self.input_file_path)
        file_processor.transform_features(self.output_file_path)

        job_ids, feature_codes, features = read_test_data_as_arrays(self.input_file_path)

        means = features.mean(axis=0)

        expected_z_scores = stats.zscore(features, axis=0, ddof=1)  # using sample standard deviation
        expected_argmaxs = np.argmax(features, axis=1)
        expected_maxs = np.amax(features, axis=1)
        expected_diffs = np.abs(expected_maxs - means[expected_argmaxs])

        line_index = 0

        with open(self.output_file_path) as f:
            next(f)

            for line in f:
                job_id, z_scores, argmax, diff = _parse_transformed_line(line)

                np.testing.assert_equal(job_id, job_ids[line_index])
                np.testing.assert_allclose(z_scores, expected_z_scores[line_index])
                np.testing.assert_equal(argmax, expected_argmaxs[line_index])
                np.testing.assert_almost_equal(diff, expected_diffs[line_index])

                line_index += 1

    def test_transform_features_some_codes(self):
        selected_codes = random.sample(self.feature_codes, k=2)

        file_processor = FileProcessor(self.input_file_path)
        file_processor.transform_features(self.output_file_path, selected_codes)

        job_ids, feature_codes, features = read_test_data_as_arrays(self.input_file_path)

        selected_indexes = []

        for code in selected_codes:
            indexes = np.where(feature_codes == code)[0].tolist()
            selected_indexes += indexes

        selected_indexes = sorted(selected_indexes)

        selected_job_ids = job_ids[selected_indexes]
        selected_features = features[selected_indexes]

        means = selected_features.mean(axis=0)

        expected_z_scores = stats.zscore(selected_features, axis=0, ddof=1)  # using sample standard deviation
        expected_argmaxs = np.argmax(selected_features, axis=1)
        expected_maxs = np.amax(selected_features, axis=1)
        expected_diffs = np.abs(expected_maxs - means[expected_argmaxs])

        line_index = 0

        with open(self.output_file_path) as f:
            next(f)

            for line in f:
                job_id, z_scores, argmax, diff = _parse_transformed_line(line)

                np.testing.assert_equal(job_id, selected_job_ids[line_index])
                np.testing.assert_allclose(z_scores, expected_z_scores[line_index])
                np.testing.assert_equal(argmax, expected_argmaxs[line_index])
                np.testing.assert_almost_equal(diff, expected_diffs[line_index])

                line_index += 1

            self.assertEqual(line_index, len(selected_indexes))

    def test_generate_statistics_all_codes(self):
        file_processor = FileProcessor(self.input_file_path)
        actual_stats = file_processor.generate_statistics()

        _, _, features = read_test_data_as_arrays(self.input_file_path)

        expected_stats = {
            "count": np.uint32(features.shape[0]),
            "mean": features.mean(axis=0),
            "std": features.std(axis=0, ddof=1),
            "max": np.amax(features, axis=0),
            "min": np.amin(features, axis=0)
        }

        for metric in expected_stats:
            np.testing.assert_allclose(actual_stats[metric], expected_stats[metric])

    def test_generate_statistics_some_codes(self):
        needed_feature_codes = random.sample(self.feature_codes, k=2)

        file_processor = FileProcessor(self.input_file_path)
        actual_stats = file_processor.generate_statistics(needed_feature_codes)

        _, feature_codes, features = read_test_data_as_arrays(self.input_file_path)

        selected_indexes = []

        for code in needed_feature_codes:
            indexes = np.where(feature_codes == code)[0].tolist()
            selected_indexes += indexes

        selected_features = features[selected_indexes]

        expected_stats = {
            "count": np.uint32(selected_features.shape[0]),
            "mean": selected_features.mean(axis=0),
            "std": selected_features.std(axis=0, ddof=1),
            "max": np.amax(selected_features, axis=0),
            "min": np.amin(selected_features, axis=0)
        }

        for metric in expected_stats:
            np.testing.assert_allclose(actual_stats[metric], expected_stats[metric])


def _parse_transformed_line(line):
    columns = line.split('\t')

    job_id = np.int64(columns[0])
    z_scores = np.array(columns[1:257], dtype=np.float64)
    argmax = np.int32(columns[257])
    diff = np.float64(columns[258])

    return job_id, z_scores, argmax, diff


if __name__ == '__main__':
    unittest.main()