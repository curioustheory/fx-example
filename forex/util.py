import json
import os

import pandas as pd
import numpy as np


def load_config(config_file_path):
    """
    load the json file into a dictionary.

    :param config_file_path: file path
    :return: dictionary
    """
    if is_file_exists(config_file_path):
        with open(file=config_file_path) as f:
            return json.loads(f.read())
    else:
        raise FileNotFoundError(config_file_path + " does not exists")


def load_data(input_file_path, input_file_header):
    """
    load the data from the file path.

    :param input_file_path: file path
    :param input_file_header: csv header
    :return: pandas dataframe
    """
    if is_file_exists(input_file_path):
        return pd.read_csv(input_file_path, names=input_file_header)
    else:
        raise FileNotFoundError(input_file_path + " does not exists")


def is_file_exists(file_path):
    """
    check if the file exist.

    :param file_path: file path
    :return:
    """
    return os.path.isfile(file_path)


def drop_highly_correlated_features(dataframe, correlation_limit=0.95):
    """
    check for highly correlated feature and drop them if exceed the defined limit.

    :param dataframe:
    :param correlation_limit:
    :return: pandas dataframe
    """
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


def split_train_test_data(X, y, train_size=0.7):
    """
    splits the data into train and test set

    :return: train and test set
    """
    print("splitting train / test set:")
    train_size = int(len(X) * train_size)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # reset index for plotting
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("Total rows: {}".format(len(X)))
    print("Training rows: {}".format(len(X_train)))
    print("Testing rows: {}".format(len(X_test)))
    print()
    return X_train, y_train, X_test, y_test
