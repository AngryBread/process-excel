import fire
import pandas as pd
import numpy as np
import os

from utils import read_csv, write_csv, split_table_by, fresh_index


g_group_key_len = 4


def save_data_to(data, src_file, out_folder, group_key, len_pad, win_size):
    if not os.path.isdir(out_folder):
        raise NotADirectoryError(out_folder)

    name, ext = os.path.splitext(os.path.basename(src_file))
    win_decorator = f'-win{win_size}'
    pad_decorator = f'-padded(l#{len_pad[0]:04}-r#{len_pad[1]:04})'
    key_decorator = f'-{group_key[-g_group_key_len:]}'
    out_file_name = name + win_decorator + key_decorator + pad_decorator + ext

    out_file_path = os.path.join(out_folder, out_file_name)
    write_csv(data, out_file_path)

    print(f'  -- write to {out_file_path}')


def filter_table(pd_data_dict: dict, black_list: list):
    return {k: d for k, d in pd_data_dict.items() if k not in black_list}


def padding(pd_data, window_size, pad_front):
    idx_times = np.ones(len(pd_data), dtype=int)
    pad_len = window_size - len(pd_data)
    idx_times[0 if pad_front else -1] += pad_len
    padded_data = pd_data.iloc[np.repeat(range(len(pd_data)), idx_times)]
    return padded_data, pad_len


def sample_at_the_valley(pd_data, window_size, key):
    # sort by the column timestamp
    # here I did not sort since the data seems already sorted

    valley_idx = np.argmin(np.array(pd_data[key]))

    lower_bound = max(valley_idx-window_size, 0)
    upper_bound = min(valley_idx+window_size, len(pd_data))

    prev_win = pd_data.iloc[lower_bound:valley_idx]
    post_win = pd_data.iloc[valley_idx:upper_bound]

    len_padding_l, len_padding_r = 0, 0
    if len(prev_win) < window_size:
        prev_win, len_padding_l = padding(prev_win, window_size, True)
    if len(post_win) < window_size:
        post_win, len_padding_r = padding(post_win, window_size, False)

    return pd.concat([prev_win, post_win]), (len_padding_l, len_padding_r)


def process_one(csv_file: str, out_folder: str, win_size: int,
                filter_blacklist: list = [], filter_key: str = 'epc',
                align_key: str = 'rssi', group_key_len: int = g_group_key_len):
    """
    Group, align and extract fix-window data sequence from a given file
    `csv_file`. Firstly, the data will be grouped by the given `filter_key` and
    filtered by `filter_blacklist`. Next, the extracted sequence will be aligned
    by the valley under key `align_key` and a fragment with a length of (2 *
    window_size) is captured. In the end, the extracted sequence will be stored
    into a file as:
      '{src_file}-win{n_win}-padded(l#{l_pad}-r#{r_pad})-{group_key[-group_key_len:]}'
    under the given output dir `out_folder`

    :param csv_file: stores the raw data sequence
    :param out_folder: where the output files will be stored
    :param win_size: the window_size (half) that will be captured
    :param filter_blacklist: the ignored key value of the group key
    :param filter_key: the key used to group the data sequence and to filter
    :param align_key: the key that used to find the alignment point
    :param group_key_len: the last `group_key_len` of the group_key will be used
    to make up the output filename
    """
    print(f'Processing file `{csv_file}`')

    global g_group_key_len
    g_group_key_len = group_key_len

    if not filter_blacklist:
        filter_blacklist = [
            'F01000310F30010011711560',
            'E2003066701700620960B90B'
        ]

    pd_data = read_csv(csv_file)
    pd_data_dict = split_table_by(pd_data, filter_key)
    pd_data_dict = filter_table(pd_data_dict, filter_blacklist)
    for key, data in pd_data_dict.items():
        data = fresh_index(data)
        sample_data, len_pad = sample_at_the_valley(data, win_size, align_key)
        save_data_to(sample_data, csv_file, out_folder, key, len_pad, win_size)


def process_all(src_folder: str, *args, **kwargs):
    for root, _, files in os.walk(src_folder):
        for file in files:
            process_one(os.path.join(root, file), *args, **kwargs)


if __name__ == '__main__':
    fire.Fire({
        'all': process_all,
        'one': process_one
    })
