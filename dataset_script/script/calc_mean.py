import re

import torch

from sklearn.preprocessing import normalize

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

from data_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir_vec_pairs = [
    (0, 1, 0.26),
    (1, 2, 0.22),
    (1, 3, 0.22),
    (2, 4, 0.36),
    (4, 6, 0.33),
    (6, 13, 0.1),
    (6, 14, 0.12),
    (6, 15, 0.14),
    (6, 16, 0.13),
    (6, 17, 0.12),

    (3, 5, 0.36),
    (5, 7, 0.33),
    (7, 18, 0.1),
    (7, 19, 0.12),
    (7, 20, 0.14),
    (7, 21, 0.13),
    (7, 22, 0.12),

    (1, 8, 0.18),
    (8, 9, 0.14),
    (8, 10, 0.14),
    (9, 11, 0.15),
    (10, 12, 0.15),
]

def calc_mean_dir():
    video_files = sorted(glob.glob(my_config.VIDEO_PATH + "/*.mp4"), key=os.path.getmtime)
    # video_files = video_files[:1]
    for v_i, video_file in enumerate(tqdm(video_files)):
        vid = os.path.split(video_file)[1][-15:-4]
        clip_data = load_clip_data(vid)
        if clip_data is None:
            print('[ERROR] clip data file does not exist!')
            break

        video_wrapper = read_video(my_config.VIDEO_PATH, vid)

        for ia, clip in enumerate(clip_data):
            # skip FALSE clips
            if not clip['clip_info'][2]:
                continue
            clip_pose_3d = clip['3d']
            for frame in clip_pose_3d[:-1]:
                if frame:
                    joints_full = frame['joints']
                    up_joints = np.vstack((joints_full[9], joints_full[12], joints_full[16:22], joints_full[55:60], joints_full[66:76])).astype('float32')


def convert_dir_vec_to_pose(vec):
    vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((23, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 23, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 23, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False

    return joint_pos

def convert_pose_seq_to_dir_vec(pose):
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        for j in range(dir_vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec

if __name__ == '__main__':
    calc_mean_dir()