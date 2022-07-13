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


def read_subtitle(vid):
    postfix_in_filename = '-en.vtt'
    file_list = glob.glob(my_config.SUBTITLE_PATH + '/*' + vid + postfix_in_filename)
    if len(file_list) > 1:
        print('more than one subtitle. check this.', file_list)
        assert False
    if len(file_list) == 1:
        return WebVTT().read(file_list[0])
    else:
        return []


# turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def normalize_subtitle(vtt_subtitle):
    for i, sub in enumerate(vtt_subtitle):
        vtt_subtitle[i].text = normalize_string(vtt_subtitle[i].text)
    return vtt_subtitle


def make_ted_gesture_dataset():
    dataset_train = []
    dataset_val = []
    dataset_test = []
    n_saved_clips = [0, 0, 0]

    out_lmdb_dir_train = my_config.OUTPUT_PATH + '/train'
    out_lmdb_dir_val = my_config.OUTPUT_PATH + '/val'
    out_lmdb_dir_test = my_config.OUTPUT_PATH + '/test'
    if not os.path.exists(out_lmdb_dir_train):
        os.makedirs(out_lmdb_dir_train)
    if not os.path.exists(out_lmdb_dir_val):
        os.makedirs(out_lmdb_dir_val)
    if not os.path.exists(out_lmdb_dir_test):
        os.makedirs(out_lmdb_dir_test)

    video_files = sorted(glob.glob(my_config.VIDEO_PATH + "/*.mp4"), key=os.path.getmtime)
    video_files = video_files[:10]
    for v_i, video_file in enumerate(tqdm(video_files)):

        vid = os.path.split(video_file)[1][-15:-4]
        # print(vid)
        audio_path = my_config.AUDIO_PATH + '/' + vid + '.mp3'
        audio_file, _ = librosa.load(audio_path, sr = 16000, mono = True)

        # load clip, video, and subtitle
        clip_data = load_clip_data(vid)
        if clip_data is None:
            print('[ERROR] clip data file does not exist!')
            break

        video_wrapper = read_video(my_config.VIDEO_PATH, vid)

        subtitle_type = my_config.SUBTITLE_TYPE
        subtitle = SubtitleWrapper(vid, subtitle_type).get()

        if subtitle is None:
            print('[WARNING] subtitle does not exist! skipping this video.')
            continue

        dataset_train.append({'vid': vid, 'clips': [], 'framerate': video_wrapper.framerate})
        dataset_val.append({'vid': vid, 'clips': [], 'framerate': video_wrapper.framerate})
        dataset_test.append({'vid': vid, 'clips': [], 'framerate': video_wrapper.framerate})

        word_index = 0
        valid_clip_count = 0
        for ia, clip in enumerate(clip_data):
            # skip FALSE clips
            if not clip['clip_info'][2]:
                continue
            
            start_frame_no, end_frame_no, clip_pose_2d, clip_pose_3d = clip['clip_info'][0], clip['clip_info'][1], clip['frames'], clip['3d']
            start_time = start_frame_no / video_wrapper.framerate
            end_time = end_frame_no / video_wrapper.framerate
            audio_start = math.floor(start_frame_no / video_wrapper.framerate * 16000)
            audio_end = math.ceil(end_frame_no / video_wrapper.framerate * 16000)
            audio_raw = audio_file[audio_start:audio_end].astype('float16')

            melspec = librosa.feature.melspectrogram(y=audio_raw, sr=16000, n_fft=1024, hop_length=512, power=2)
            log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
            audio_feat = log_melspec.astype('float16')
            # print(audio_raw.shape)
            # print(audio_feat.shape)

            clip_word_list = []

            # train/val/test split
            if valid_clip_count % 10 == 9:
                dataset = dataset_test
                dataset_idx = 2
            elif valid_clip_count % 10 == 8:
                dataset = dataset_val
                dataset_idx = 1
            else:
                dataset = dataset_train
                dataset_idx = 0
            valid_clip_count += 1

            # get subtitle that fits clip
            for ib in range(word_index - 1, len(subtitle)):
                if ib < 0:
                    continue

                word_s = video_wrapper.second2frame(subtitle[ib]['start'])
                word_e = video_wrapper.second2frame(subtitle[ib]['end'])
                word = subtitle[ib]['word']

                if word_s >= end_frame_no:
                    word_index = ib
                    break

                if word_e <= start_frame_no:
                    continue

                word = normalize_string(word)
                clip_word_list.append([word, word_s / video_wrapper.framerate, word_e / video_wrapper.framerate])

            if clip_word_list:
                clip_skeleton_2d = []
                clip_skeleton_3d = []

                # get skeletons of the upper body in the clip
                for frame in clip_pose_2d:
                    if frame:
                        clip_skeleton_2d.append(get_skeleton_from_frame(frame)[:24])
                    else:  # frame with no skeleton
                        clip_skeleton_2d.append([0] * 24)

                for frame in clip_pose_3d[:-1]:
                    if frame:
                        joints_full = frame['joints']
                        up_joints = np.vstack((joints_full[9], joints_full[12], joints_full[16:22], joints_full[55:60], joints_full[66:76])).astype('float32')
                        clip_skeleton_3d.append(up_joints)
                    else:  # frame with no skeleton
                        clip_skeleton_3d.append(np.zeros((23, 3), dtype = float32))

                # proceed if skeleton list is not empty
                if len(clip_skeleton_2d) > 0:
                    # save subtitles and skeletons corresponding to clips
                    n_saved_clips[dataset_idx] += 1
                    dataset[-1]['clips'].append({'words': clip_word_list,
                                                 'skeletons': clip_skeleton_2d,
                                                 'skeletons_3d': clip_skeleton_3d,
                                                 'audio_feat': audio_feat,
                                                 'audio_raw': audio_raw,
                                                 'start_frame_no': start_frame_no, 
                                                 'end_frame_no': end_frame_no,
                                                 'start_time': start_time,
                                                 'end_time': end_time
                                                 })
                    print('{} ({}, {})'.format(vid, start_frame_no, end_frame_no))
                else:
                    print('{} ({}, {}) - consecutive missing frames'.format(vid, start_frame_no, end_frame_no))

    # for debugging
    # if vid == 'yq3TQoMjXTw':
    #     break

    print('writing to pickle...')
    with open(out_lmdb_dir_train + 'ted_expressive_dataset_train.pickle', 'wb') as f:
        pickle.dump(dataset_train, f)
    with open(out_lmdb_dir_train + 'ted_expressive_dataset_val.pickle', 'wb') as f:
        pickle.dump(dataset_val, f)
    with open(out_lmdb_dir_train + 'ted_expressive_dataset_test.pickle', 'wb') as f:
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

    print('no. of saved clips: train {}, val {}, test {}'.format(n_saved_clips[0], n_saved_clips[1], n_saved_clips[2]))


if __name__ == '__main__':
    make_ted_gesture_dataset()
