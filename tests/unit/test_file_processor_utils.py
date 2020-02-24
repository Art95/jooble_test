import unittest
import numpy as np
import file_processor.utils as utils

class TestFileProcessorUtils(unittest.TestCase):
    def test_parse_line(self):
        line = "-12345\t10,2345,3456,4567,5678,6789"
        job_id, feature_code, features = utils.parse_line(line)

        expected_job_id = np.int64(-12345)
        expected_feature_code = np.int32(10)
        expected_features = np.array([2345,3456,4567,5678,6789], dtype=np.int32)

        np.testing.assert_equal(job_id, expected_job_id)
        np.testing.assert_equal(feature_code, expected_feature_code)
        np.testing.assert_array_equal(features, expected_features)

    def test_generate_line(self):
        job_id = np.int64(9999)
        transformed_features = {
            "z_scores": np.array([-0.672, 5., 1.184, -4.33], dtype=np.float64),
            "argmax": np.int32(1),
            "abs_max_mean_diff": np.float32(3.3)
        }

        line = utils.generate_line(job_id, transformed_features)
        expected_line = "9999\t-0.672\t5.0\t1.184\t-4.33\t1\t3.3\n"

        self.assertEqual(line, expected_line)

    def test_generate_column_names_line(self):
        features_size = 2

        line = utils.generate_column_names_line(features_size)
        expected_line = "job_id\tz_score_feature_0\tz_score_feature_1\targmax\tabs_max_mean_diff\n"

        self.assertEqual(line, expected_line)


if __name__ == '__main__':
    unittest.main()
