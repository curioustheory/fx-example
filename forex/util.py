import json
import os

import pandas as pd


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
