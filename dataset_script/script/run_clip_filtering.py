# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

from __future__ import unicode_literals
import csv
from clip_filter import *
from main_speaker_selector import *
from config import my_config
import numpy as np

RESUME_VID = ''  # resume the process from this video


def read_sceneinfo(filepath):  # reading csv file
    with open(filepath, 'r') as csv_file:
        frame_list = [0]
        for row in csv.reader(csv_file):
            if row:
                frame_list.append((row[1]))
        frame_list[0:3] = []  # skip header

    frame_list = [int(x) for x in frame_list]  # str to int

    return frame_list


def run_filtering(vid, scene_data, skeleton_wrapper, video_wrapper):
    filtered_clip_data = []
    aux_info = []
    video = video_wrapper.get_video_reader()
    height = video_wrapper.height
    width = video_wrapper.width

    for i in range(len(scene_data) - 1):  # note: last scene is not processed
        start_frame_no, end_frame_no = scene_data[i], scene_data[i + 1]
        raw_skeleton_chunk = skeleton_wrapper.get(start_frame_no, end_frame_no)
        main_speaker_skeletons = MainSpeakerSelector(raw_skeleton_chunk=raw_skeleton_chunk).get()

        # run clip filtering
        clip_filter = ClipFilter(vid = vid, video=video, start_frame_no=start_frame_no, end_frame_no=end_frame_no,
                                 raw_skeleton=raw_skeleton_chunk, main_speaker_skeletons=main_speaker_skeletons, height = height, width = width)
        correct_clip = clip_filter.is_correct_clip()

        filtering_results, message, debugging_info = clip_filter.get_filter_variable()
        filter_elem = {'clip_info': [start_frame_no, end_frame_no, correct_clip], 'filtering_results': filtering_results,
                       'message': message, 'debugging_info': debugging_info}
        aux_info.append(filter_elem)

        # save
        elem = {'clip_info': [start_frame_no, end_frame_no, correct_clip], 'frames': [], '3d': []}

        if not correct_clip:
            filtered_clip_data.append(elem)
            continue
        elem['frames'] = main_speaker_skeletons
        expose_list = []
        for ii in range(start_frame_no, end_frame_no + 1):
            expose_path = my_config.EXPOSE_OUT_PATH + '/' + vid + '/' + '%05d'%ii + '.npz'
            if (not os.path.exists(expose_path)):
                expose_list.append({})
                continue
            else:
                new_dict = {}
                file = np.load(expose_path)
                for key in file.files:
                    if key not in ['fname', 'full_pose']:
                        if isinstance(file[key], (np.ndarray)):
                            new_dict[key] = file[key].tolist()
                        else:
                            new_dict[key] = file[key]
                expose_list.append(new_dict)
        elem['3d'] = expose_list
        filtered_clip_data.append(elem)

    return filtered_clip_data, aux_info


def main():
    if RESUME_VID == "":
        skip_flag = False
    else:
        skip_flag = True

    file_list = sorted(glob.glob(my_config.CLIP_PATH + "/*.csv"), key=os.path.getmtime)
    # file_list = file_list[:1]

    for csv_path in file_list:

        vid = os.path.split(csv_path)[1][0:11]
        print(vid)

        # resume check
        if skip_flag and vid == RESUME_VID:
            skip_flag = False

        if not skip_flag:
            scene_data = read_sceneinfo(csv_path)
            skeleton_wrapper = SkeletonWrapper(my_config.SKELETON_PATH, vid)
            video_wrapper = read_video(my_config.VIDEO_PATH, vid)

            # if video_wrapper.height < 720:
            #     print('[Fatal error] wrong video size (height: {})'.format(video_wrapper.height))
            #     assert False

            if abs(video_wrapper.total_frames - len(skeleton_wrapper.skeletons)) > 10:
                print('[Fatal error] video and skeleton object have different lengths (video: {}, skeletons: {})'.format
                      (video_wrapper.total_frames, len(skeleton_wrapper.skeletons)))
                assert False

            if skeleton_wrapper.skeletons == [] or video_wrapper is None:
                print('[warning] no skeleton or video! skipped this video.')
            else:
                ###############################################################################################
                filtered_clip_data, aux_info = run_filtering(vid, scene_data, skeleton_wrapper, video_wrapper)
                ###############################################################################################

                # save filtered clips and aux info
                with open("{}/{}.json".format(my_config.FILTER_PATH, vid), 'w') as clip_file:
                    json.dump(filtered_clip_data, clip_file)
                with open("{}/{}_aux_info.json".format(my_config.FILTER_PATH, vid), 'w') as aux_file:
                    json.dump(aux_info, aux_file)


if __name__ == '__main__':
    main()
