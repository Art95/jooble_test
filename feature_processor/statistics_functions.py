import numpy as np

def update_statistics(statistics, features):
    new_count = statistics["count"] + 1
    new_mean = _calculate_dynamic_mean(features, statistics["mean"], new_count)

    updated_stats = {
        "count": new_count,
        "mean": new_mean,
        "std": _calculate_dynamic_sample_std(features, statistics["std"], statistics["mean"],
                                             new_mean, new_count),
        "max": np.maximum(statistics["max"], features),
        "min": np.minimum(statistics["min"], features)
    }

    return updated_stats

def merge_statistics(statistics):
    if len(statistics) == 0:
        return statistics

    counts = np.array([stat["count"] for stat in statistics])
    means = np.array([stat["mean"] for stat in statistics])
    stds = np.array([np.nan_to_num(stat["std"]) for stat in statistics])
    mins = np.array([stat["min"] for stat in statistics])
    maxs = np.array([stat["max"] for stat in statistics])

    counts = counts.reshape((counts.size, 1))

    merged_counts = np.sum(counts)
    merged_means = np.sum(counts * means, axis=0) / merged_counts
    merged_stds = np.sqrt((np.sum(np.subtract(counts, 1) * stds ** 2 + counts * means ** 2, axis=0) - merged_counts * merged_means ** 2) / (merged_counts - 1))

    merged_stats = {
        "count": merged_counts,
        "mean": merged_means,
        "std": merged_stds,
        "max": np.amax(maxs, axis=0),
        "min": np.amin(mins, axis=0)
    }

    return merged_stats

def _calculate_dynamic_mean(features, old_means, new_count):
    vec_mean = np.vectorize(_calc_mean)
    return vec_mean(features, old_means, new_count)

def _calculate_dynamic_sample_std(features, old_stds, old_mean, new_mean, new_count):
    vec_std = np.vectorize(_calc_std)
    return vec_std(features, old_stds, old_mean, new_mean, new_count)

def _calc_mean(feature, mean, count):
    return mean + (feature - mean) / count

def _calc_std(feature, std, old_mean, new_mean, new_count):
    n = new_count

    if n == 1:
        return np.nan

    var = std ** 2 if not np.isnan(std) else 0
    new_var = ((n - 2) * var + (n - 1) * (new_mean - old_mean) ** 2 + (feature - new_mean) ** 2) / (n - 1)

    return np.sqrt(new_var)

