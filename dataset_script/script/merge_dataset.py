# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------
import glob
import os
import pickle
import sys

import cv2
import math
import lmdb
import numpy as np
from numpy import float32
from tqdm import tqdm

import unicodedata
import librosa
import pyarrow

from data_utils import *

def merge_dataset():
    dataset_train = []
    dataset_val = []
    dataset_test = []
    pickle_folder_list = os.listdir(my_config.OUTPUT_PATH)

    out_lmdb_dir_train = my_config.OUTPUT_PATH + '/train'
    out_lmdb_dir_val = my_config.OUTPUT_PATH + '/val'
    out_lmdb_dir_test = my_config.OUTPUT_PATH + '/test'
    if not os.path.exists(out_lmdb_dir_train):
        os.makedirs(out_lmdb_dir_train)
    if not os.path.exists(out_lmdb_dir_val):
        os.makedirs(out_lmdb_dir_val)
    if not os.path.exists(out_lmdb_dir_test):
        os.makedirs(out_lmdb_dir_test)

    for dir in pickle_folder_list:
        pickle_file = my_config.OUTPUT_PATH + '/' + dir + '/ted_expressive_dataset_train.pickle'
        with open(pickle_file, 'rb') as file:
            temp_train = pickle.load(file)
        dataset_train.extend(temp_train)

    for dir in pickle_folder_list:
        pickle_file = my_config.OUTPUT_PATH + '/' + dir + '/ted_expressive_dataset_val.pickle'
        with open(pickle_file, 'rb') as file:
            temp_val = pickle.load(file)
        dataset_val.extend(temp_val)

    for dir in pickle_folder_list:
        pickle_file = my_config.OUTPUT_PATH + '/' + dir + '/ted_expressive_dataset_test.pickle'
        with open(pickle_file, 'rb') as file:
            temp_test = pickle.load(file)
        dataset_test.extend(temp_test)

    print('writing to pickle...')
    with open(my_config.OUTPUT_PATH + '/' + 'ted_expressive_dataset_train.pickle', 'wb') as f:
        pickle.dump(dataset_train, f)
    with open(my_config.OUTPUT_PATH + '/' + 'ted_expressive_dataset_val.pickle', 'wb') as f:
        pickle.dump(dataset_val, f)
    with open(my_config.OUTPUT_PATH + '/' + 'ted_expressive_dataset_test.pickle', 'wb') as f:
        pickle.dump(dataset_test, f)

    map_size = 1024 * 100  # in MB
    map_size <<= 20  # in B
    env_train = lmdb.open(out_lmdb_dir_train, map_size=map_size)
    env_val = lmdb.open(out_lmdb_dir_val, map_size=map_size)
    env_test = lmdb.open(out_lmdb_dir_test, map_size=map_size)

    # lmdb train
    with env_train.begin(write=True) as txn:
        for idx, dic in enumerate(dataset_train):
            k = '{:010}'.format(idx).encode('ascii')
            v = pyarrow.serialize(dic).to_buffer()
            txn.put(k, v)
    env_train.close()

    # lmdb val
    with env_val.begin(write=True) as txn:
        for idx, dic in enumerate(dataset_val):
            k = '{:010}'.format(idx).encode('ascii')
            v = pyarrow.serialize(dic).to_buffer()
            txn.put(k, v)
    env_val.close()

    # lmdb test
    with env_test.begin(write=True) as txn:
        for idx, dic in enumerate(dataset_test):
            k = '{:010}'.format(idx).encode('ascii')
            v = pyarrow.serialize(dic).to_buffer()
            txn.put(k, v)
    env_test.close()


if __name__ == '__main__':
    merge_dataset()
