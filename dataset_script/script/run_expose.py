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

# python inference.py --exp-cfg data/conf.yaml --datasets openpose --exp-opts datasets.body.batch_size 1 datasets.body.openpose.data_folder /mnt/lustre/liuxian/youtube-gesture-dataset/temp_skeleton_raw/-2Dj9M71JAc --show False --output-folder OUTPUT_FOLDER --save-params True --save-vis False --save-mesh False
# python inference.py --exp-cfg data/conf.yaml --datasets openpose --exp-opts datasets.body.batch_size 64 datasets.body.openpose.data_folder /home/yuxi/openpose/youtube-gesture-dataset/temp_skeleton_raw/-2Dj9M71JAc --show False --output-folder OUTPUT_FOLDER --save-params True --save-vis False --save-mesh False
# python demo.py --image-folder samples --exp-cfg data/conf.yaml --show=False --output-folder OUTPUT_FOLDER --save-params True --save-vis False --save-mesh False

import glob
import json
import os
import pickle
import subprocess

import shutil

from config import my_config

# maximum accuracy, too slow (~1fps)
# OPENPOSE_OPTION = "--net_resolution -1x736 --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --face"
# OPENPOSE_OPTION = "--face --hand --number_people_max 1 -model_pose COCO --display 0 --render_pose 0"

OUTPUT_SKELETON_PATH = my_config.WORK_PATH + "/temp_skeleton_raw"
OUTPUT_3D_PATH = my_config.WORK_PATH + "/expose_ted"

RESUME_VID = ""  # resume from this video
SKIP_EXISTING_SKELETON = True  # skip if the skeleton file is existing


def get_vid_from_filename(filename):
    return filename[-15:-4]


def read_skeleton_json(_file):
    with open(_file) as json_file:
        skeleton_json = json.load(json_file)
        return skeleton_json['people']


def save_skeleton_to_pickle(_vid):
    files = glob.glob(OUTPUT_SKELETON_PATH + '/' + _vid + '/*.json')
    if len(files) > 10:
        files = sorted(files)
        skeletons = []
        for file in files:
            skeletons.append(read_skeleton_json(file))
        with open(my_config.SKELETON_PATH + '/' + _vid + '.pickle', 'wb') as file:
            pickle.dump(skeletons, file)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_3D_PATH):
        os.makedirs(OUTPUT_3D_PATH)
    
    os.chdir(my_config.EXPOSE_BASE_DIR)

    if RESUME_VID == "":
        skip_flag = False
    else:
        skip_flag = True

    video_files = glob.glob(my_config.VIDEO_PATH + "/*.mp4")
    
    for file in sorted(video_files, key=os.path.getmtime):
        # print(file)
        vid = get_vid_from_filename(file)
        print(vid)

        skip_iter = False

        # resume check
        if skip_flag and vid == RESUME_VID:
            skip_flag = False
        skip_iter = skip_flag

        # # existing skeleton check
        # if SKIP_EXISTING_SKELETON:
        #     if os.path.exists(my_config.SKELETON_PATH + '/' + vid + '.pickle'):
        #         print('existing skeleton')
        #         skip_iter = True

        if not skip_iter:
            # create out dir
            expose_out = OUTPUT_3D_PATH + "/" + vid
            if os.path.exists(expose_out):
                shutil.rmtree(expose_out)
            
            os.makedirs(expose_out)

            # call expose
            command = "python " + my_config.EXPOSE_BASE_DIR + "inference.py --exp-cfg " + my_config.EXPOSE_BASE_DIR + "data/conf.yaml --datasets openpose --exp-opts datasets.body.batch_size 256 datasets.body.openpose.data_folder " + OUTPUT_SKELETON_PATH + "/" + vid + " --show False --output-folder " + expose_out + " --save-params True --save-vis False --save-mesh False"
            print(command)
            subprocess.call(command, shell=True)

            # save skeletons to a pickle file
            # save_skeleton_to_pickle(vid)