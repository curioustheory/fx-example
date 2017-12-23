import json

import pandas as pd


def load_config(config_file_path):
    with open(file=config_file_path) as f:
        return json.loads(f.read())


def load_data(input_file_path, input_file_header):
    return pd.read_csv(input_file_path, names=input_file_header)


def _file_exist(config_file_path):
    pass
