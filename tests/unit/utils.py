import numpy as np

def read_test_data(file_path):
    test_data = {}
    unique_feature_codes = set()

    with open(file_path) as data_file:
        for line in data_file:
            columns = line.split('\t')
            values = columns[1].split(',')

            job_id = np.int64(columns[0])
            feature_code = np.int32(values[0])
            features = np.array(values[1:], dtype=np.int32)

            if job_id not in test_data:
                test_data[job_id] = {}

            test_data[job_id][feature_code] = features
            unique_feature_codes.add(feature_code)

    feature_codes = list(unique_feature_codes)

    return test_data, feature_codes

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
                if filtered_features.size else features

    return filtered_features