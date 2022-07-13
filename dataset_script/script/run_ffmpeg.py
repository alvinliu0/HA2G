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

OUTPUT_SKELETON_PATH = my_config.WORK_PATH + "/temp_skeleton_raw"

def get_vid_from_filename(filename):
    return filename[-15:-4]


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_SKELETON_PATH):
        os.makedirs(OUTPUT_SKELETON_PATH)

    video_files = glob.glob(my_config.VIDEO_PATH + "/*.mp4")
    for file in sorted(video_files, key=os.path.getmtime):
        print(file)
        vid = get_vid_from_filename(file)
        print(vid)

        # create out dir
        skeleton_dir = OUTPUT_SKELETON_PATH + "/" + vid + "/"
        if not os.path.exists(skeleton_dir):
            os.makedirs(skeleton_dir)

        if os.path.exists(skeleton_dir + "images/"):
            shutil.rmtree(skeleton_dir + "images/")

        os.makedirs(skeleton_dir + "images/")

        command_ffmpeg = "ffmpeg -i " + my_config.VIDEO_PATH + "/" + vid + ".mp4 -start_number 0 -f image2 " + OUTPUT_SKELETON_PATH + "/" + vid + "/images/" + vid + "_%012d" + ".png"
        print(command_ffmpeg)
        subprocess.call(command_ffmpeg, shell=True)

# import glob
# import json
# import os
# import pickle
# import subprocess

# import shutil

# def main():
#     lmk_root = '/mnt/lustressd/share/wangyuxin1/DATASETS/lrw-v1/lipread_ldmk/'
#     align_root = '/mnt/lustre/DATAshare3/lrw_srcM/'
#     local_root_lmk = '/mnt/lustre/liuxian.vendor/lrw_lmk/'
#     local_root_align = '/mnt/lustre/liuxian.vendor/lrw_align/'
#     category_list = os.listdir(lmk_root)
#     for category in category_list:
#         if os.path.exists(local_root_lmk + category):
#             shutil.rmtree(local_root_lmk + category)
#         os.makedirs(local_root_lmk + category)
#         abs_train = local_root_lmk + category + '/train'
#         abs_val = local_root_lmk + category + '/val'
#         abs_test = local_root_lmk + category + '/test'
#         if os.path.exists(abs_train):
#             shutil.rmtree(abs_train)
#         if os.path.exists(abs_val):
#             shutil.rmtree(abs_val)
#         if os.path.exists(abs_test):
#             shutil.rmtree(abs_test)
#         os.makedirs(abs_train)
#         os.makedirs(abs_val)
#         os.makedirs(abs_test)

#         fname_list_train = os.listdir(lmk_root + category + '/train/')
#         fname_list_val = os.listdir(lmk_root + category + '/val/')
#         fname_list_test = os.listdir(lmk_root + category + '/test/')

#         for fname in fname_list_train:
#             if os.path.exists(abs_train + '/' + fname):
#                 shutil.rmtree(abs_train + '/' + fname)
#             os.makedirs(abs_train + '/' + fname)
#             command = "cp " + lmk_root + category + '/train/' + fname + '/lmk.txt ' + abs_train + '/' + fname + '/lmk.txt'
#             subprocess.call(command, shell=True)
        
#         for fname in fname_list_val:
#             if os.path.exists(abs_val + '/' + fname):
#                 shutil.rmtree(abs_val + '/' + fname)
#             os.makedirs(abs_val + '/' + fname)
#             command = "cp " + lmk_root + category + '/val/' + fname + '/lmk.txt ' + abs_val + '/' + fname + '/lmk.txt'
#             subprocess.call(command, shell=True)

#         for fname in fname_list_test:
#             if os.path.exists(abs_test + '/' + fname):
#                 shutil.rmtree(abs_test + '/' + fname)
#             os.makedirs(abs_test + '/' + fname)
#             command = "cp " + lmk_root + category + '/test/' + fname + '/lmk.txt ' + abs_test + '/' + fname + '/lmk.txt'
#             subprocess.call(command, shell=True)
            
#     category_list = os.listdir(align_root)
#     for category in category_list:
#         if os.path.exists(local_root_align + category):
#             shutil.rmtree(local_root_align + category)
#         os.makedirs(local_root_align + category)
#         abs_train = local_root_align + category + '/train'
#         abs_test = local_root_align + category + '/test'
#         if os.path.exists(abs_train):
#             shutil.rmtree(abs_train)
#         if os.path.exists(abs_test):
#             shutil.rmtree(abs_test)
#         os.makedirs(abs_train)
#         os.makedirs(abs_test)

#         fname_list_train = glob.glob(align_root + category + '/train/*.txt')
#         fname_list_test = glob.glob(align_root + category + '/test/*.txt')

#         for fname in fname_list_train:
#             fname = fname[:-4]
#             if os.path.exists(abs_train + '/' + fname):
#                 shutil.rmtree(abs_train + '/' + fname)
#             os.makedirs(abs_train + '/' + fname)
#             command = "cp " + align_root + category + '/train/' + fname + '.txt ' + abs_train + '/' + fname + '/align.txt'
#             subprocess.call(command, shell=True)
        
#         for fname in fname_list_test:
#             fname = fname[:-4]
#             if os.path.exists(abs_test + '/' + fname):
#                 shutil.rmtree(abs_test + '/' + fname)
#             os.makedirs(abs_test + '/' + fname)
#             command = "cp " + align_root + category + '/test/' + fname + '.txt ' + abs_test + '/' + fname + '/align.txt'
#             subprocess.call(command, shell=True)
            
# if __name__ == '__main__':
#     main()