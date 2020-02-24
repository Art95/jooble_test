import numpy as np

def parse_line(line):
    columns = line.split('\t')
    values = columns[1].split(',')

    job_id = np.int64(columns[0])
    feature_code = np.int32(values[0])
    features = np.array(values[1:], dtype=np.int32)

    return job_id, feature_code, features


def generate_line(job_id, transformed_features):
    line = str(job_id)
    line += "\t" + "\t".join(map(str, transformed_features["z_scores"].tolist()))
    line += "\t" + np.str(transformed_features["argmax"])
    line += "\t" + np.str(transformed_features["abs_max_mean_diff"])
    line += "\n"

    return line


def generate_column_names_line(features_size):
    z_score_columns = ["z_score_feature_{}".format(i) for i in range(features_size)]

    line = "job_id"
    line += "\t" + "\t".join(z_score_columns)
    line += "\t" + "argmax"
    line += "\t" + "abs_max_mean_diff"
    line += "\n"

    return line
