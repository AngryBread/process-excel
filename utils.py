import os
import pandas as pd


def read_csv(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(csv_file)
    return pd.read_csv(csv_file, index_col=0)


def write_csv(pd_data, csv_file, force=True):
    if not force and os.path.exists(csv_file):
        raise FileExistsError(f'File `{csv_file}` already exists!')
    return pd_data.to_csv(csv_file)


def split_table_by(pd_data, key):
    return {k: d for k, d in pd_data.groupby(key)}


def fresh_index(pd_data):
    return pd_data.set_index(pd.Int64Index(range(len(pd_data))))
