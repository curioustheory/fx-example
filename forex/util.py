import json
import os

import pandas as pd
import numpy as np


def load_config(config_file_path):
    if is_file_exists(config_file_path):
        with open(file=config_file_path) as f:
            return json.loads(f.read())
    else:
        raise FileNotFoundError(config_file_path + " does not exists")


def load_data(input_file_path, input_file_header):
    if is_file_exists(input_file_path):
        return pd.read_csv(input_file_path, names=input_file_header)
    else:
        raise FileNotFoundError(input_file_path + " does not exists")


def is_file_exists(file_path):
    return os.path.isfile(file_path)


def drop_highly_correlated_features(dataframe, correlation_limit=0.95):
    # Create correlation matrix
    corr_matrix = dataframe.corr().abs()

    # Select upper triangle of correlation matrix
    upper_corr_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than the limit
    columns_to_drop = [column for column in upper_corr_triangle.columns if
                       any(upper_corr_triangle[column] > correlation_limit)]

    print("drop highly correlated columns (corr > {}): {}".format(correlation_limit, str(columns_to_drop)))
    print()
    return dataframe.drop(columns_to_drop, axis=1)
