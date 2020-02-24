import numpy as np


def calculate_z_score(features, means, stds):
    vec_z_score = np.vectorize(_z_score)
    return vec_z_score(features, means, stds)


def get_argmax(features):
    return np.argmax(features)


def calculate_abs_max_mean_diff(features, means):
    index_max = get_argmax(features)
    return np.abs(features[index_max] - means[index_max])


def get_min(features):
    return np.amin(features)


def get_max(features):
    return np.amax(features)


def transform_features(features, statistics):
    z_scores = calculate_z_score(features, statistics["mean"], statistics["std"])
    argmax = get_argmax(features)
    abs_max_mean_diff = calculate_abs_max_mean_diff(features, statistics["mean"])

    transformed_features = {
        "z_scores": z_scores,
        "argmax": argmax,
        "abs_max_mean_diff": abs_max_mean_diff
    }

    return transformed_features


def _z_score(feature, mean, std):
    if std == 0:
        raise RuntimeError("Cannot calculate z-score. Standard deviation is 0.")

    return (feature - mean) / std
