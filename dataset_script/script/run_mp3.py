# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

"""
Extract pose skeletons by using OpenPose library
Need proper LD_LIBRARY_PATH before run this script
Pycharm: In RUN > Edit Configurations, add LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
"""

import glob
import json
import os
import pickle
import subprocess

import shutil

from config import my_config

def get_vid_from_filename(filename):
    return filename[-15:-4]


if __name__ == '__main__':
    audiopath = my_config.WORK_PATH + '/audio_ted'
    if not os.path.exists(audiopath):
        os.makedirs(audiopath)

    video_files = glob.glob(my_config.VIDEO_PATH + "/*.mp4")
    for file in sorted(video_files, key=os.path.getmtime):
        print(file)
        vid = get_vid_from_filename(file)
        print(vid)

        command = "ffmpeg -i " + my_config.VIDEO_PATH + "/" + vid + ".mp4 " + audiopath + '/' + vid + ".mp3"
        print(command)
        subprocess.call(command, shell=True)