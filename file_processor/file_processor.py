import numpy as np
import feature_processor
import file_processor.utils as utils


class FileProcessor:
    def __init__(self, file_path):
        self.__input_file_path = file_path
        self.__features_size = self._get_features_size()

        self.__default_stats_for_code = {
            "count": np.uint32(0),
            "mean": np.zeros(self.__features_size),
            "std": np.zeros(self.__features_size),
            "min": np.array([np.iinfo(np.int32).max] * self.__features_size),
            "max": np.array([np.iinfo(np.int32).min] * self.__features_size)
        }

        self.__global_stats = self._generate_global_statistics(file_path)

    def transform_features(self, output_file_path, feature_codes=None):
        stats = self._merge_statistics_for_codes(feature_codes)

        out_file = open(output_file_path, 'w')

        column_names = utils.generate_column_names_line(self.__features_size)
        out_file.write(column_names)

        with open(self.__input_file_path) as f:
            next(f)  # skip line with column names

            for line in f:
                job_id, feature_code, features = utils.parse_line(line)

                if feature_codes and feature_code not in feature_codes:
                    continue

                transformed_features = feature_processor.transform_features(features, stats)

                out_line = utils.generate_line(job_id, transformed_features)
                out_file.write(out_line)

        out_file.close()

    def generate_statistics(self, feature_codes=None):
        self.__global_stats = self._generate_global_statistics(self.__input_file_path)
        statistics_for_codes = self._merge_statistics_for_codes(feature_codes)

        return statistics_for_codes

    def _generate_global_statistics(self, feature_file_path):
        stats = {}

        with open(feature_file_path) as f:
            next(f)  # skip line with column names

            for line in f:
                job_id, feature_code, features = utils.parse_line(line)

                stats_for_code = stats.get(feature_code, self.__default_stats_for_code)

                updated_stats = feature_processor.update_statistics(stats_for_code, features)
                stats[feature_code] = updated_stats

            return stats

    def _merge_statistics_for_codes(self, feature_codes=None):
        if not feature_codes:
            feature_codes = list(self.__global_stats.keys())

        stats_for_codes = [self.__global_stats[code] for code in feature_codes]
        merged_stats = feature_processor.merge_statistics(stats_for_codes)

        return merged_stats

    def _get_features_size(self):
        with open(self.__input_file_path) as f:
            next(f)  # skip line with column names

            line = f.readline()

            _, _, features = utils.parse_line(line)

            return features.size
