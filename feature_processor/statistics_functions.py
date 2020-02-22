def update_statistics(statistics, features):
    raise NotImplementedError()

def merge_statistics(statistics):
    raise NotImplementedError()

def _calculate_dynamic_mean(features, old_means):
    raise NotImplementedError()

def _calculate_dynamic_sample_std(features, old_stds):
    raise NotImplementedError()