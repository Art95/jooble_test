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

def _z_score(feature, mean, std):
    if std == 0:
        raise RuntimeError("Cannot calculate z-score. Standard deviation is 0.")

    return (feature - mean) / std