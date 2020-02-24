import numpy as np

def read_test_data(file_path):
    test_data = {}
    unique_feature_codes = set()

    with open(file_path) as data_file:
        next(data_file) # skip column names

        for line in data_file:
            job_id, feature_code, features = _parse_line(line)

            if job_id not in test_data:
                test_data[job_id] = {}

            test_data[job_id][feature_code] = features
            unique_feature_codes.add(feature_code)

    feature_codes = list(unique_feature_codes)

    return test_data, feature_codes

def read_test_data_as_arrays(file_path):
    job_ids = []
    feature_codes = []
    all_features = np.array([], dtype=np.int32)

    with open(file_path) as data_file:
        next(data_file) # skip column names

        for line in data_file:
            job_id, feature_code, features = _parse_line(line)

            job_ids += [job_id]
            feature_codes += [feature_code]
            all_features = np.vstack([all_features, features]) \
                if all_features.size else np.reshape(features, (1, features.size))

        return np.array(job_ids, dtype=np.int64), np.array(feature_codes, dtype=np.int32), all_features

def filter_features_for_codes(test_data, feature_codes):
    filtered_features = np.array([])

    for feature_code in feature_codes:
        features_for_code = filter_features_for_code(test_data, feature_code)
        filtered_features = np.vstack([filtered_features, features_for_code]) \
            if filtered_features.size else features_for_code

    return filtered_features


def filter_features_for_code(test_data, feature_code):
    filtered_features = np.array([])

    for job_id in test_data:
        if feature_code in test_data[job_id]:
            features = test_data[job_id][feature_code]
            filtered_features = np.vstack([filtered_features, features]) \
                if filtered_features.size else np.reshape(features, [1, features.shape[0]])

    return filtered_features

def _parse_line(line):
    columns = line.split('\t')
    values = columns[1].split(',')

    job_id = np.int64(columns[0])
    feature_code = np.int32(values[0])
    features = np.array(values[1:], dtype=np.int32)

    return job_id, feature_code, features
