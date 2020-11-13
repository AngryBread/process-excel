import os
import re

import fire
import pandas as pd

from utils import read_csv, write_csv


def is_not_padded(file):
    return re.match(r'.*\(l#0+-r#0+\).*', file) is not None


def compute_avg_duration(src_folder, out_folder):
    durations = pd.DataFrame()
    for root, _, files in os.walk(src_folder):
        for file in files:
            # compute duration
            pd_data = read_csv(os.path.join(root, file))
            pd_data['duration'] = pd_data.time_stamp.diff().fillna(0) 
            pd_data.time_stamp -= pd_data.time_stamp.array[0]
            write_csv(pd_data, os.path.join(out_folder, file))

            # accumulate only non-pad duration
            if is_not_padded(file):
                durations = pd.concat(
                    [durations, pd_data.reset_index().duration], axis=1)
    return durations.mean(axis=1).array


def dispatch_avg_duration(folder, avg_duration_data):
    for root, _, files in os.walk(folder):
        for file in files:
            pd_data = read_csv(os.path.join(root, file))
            pd_data['avg_duration'] = avg_duration_data
            write_csv(pd_data, os.path.join(root, file), True)


def patch_duration(src_folder, out_folder):
    """
    Compute the durations, average the valid ones and then append to each table
    :param src_folder: contains sampled aligned data
    :param out_folder: where for storing the data patched with avg_duration
    """
    avg_duration_data = compute_avg_duration(src_folder, out_folder)
    dispatch_avg_duration(out_folder, avg_duration_data)


if __name__ == '__main__':
    fire.Fire(patch_duration)
