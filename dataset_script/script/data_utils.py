# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

import glob
import matplotlib
import cv2
import re
import json
import _pickle as pickle
from webvtt import WebVTT
from config import my_config

COLOR = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

## with detailed fingers
pairs_complex = [
    (0, 1, 'r'),
    (1, 2, 'r'),
    (1, 3, 'r'),
    (2, 5, 'r'),
    (5, 7, 'r'),
    (7, 9, 'r'),
    (9, 14, 'r'),
    (14, 15, 'r'),
    (15, 16, 'r'),
    (9, 17, 'r'),
    (17, 18, 'r'),
    (18, 19, 'r'),
    (9, 20, 'r'),
    (20, 21, 'r'),
    (21, 22, 'r'),
    (9, 23, 'r'),
    (23, 24, 'r'),
    (24, 25, 'r'),
    (9, 26, 'r'),
    (26, 27, 'r'),
    (27, 28, 'r'),
    (3, 6, 'r'),
    (6, 8, 'r'),
    (8, 10, 'r'),
    (10, 29, 'r'),
    (29, 30, 'r'),
    (30, 31, 'r'),
    (10, 32, 'r'),
    (32, 33, 'r'),
    (33, 34, 'r'),
    (10, 35, 'r'),
    (35, 36, 'r'),
    (36, 37, 'r'),
    (10, 38, 'r'),
    (38, 39, 'r'),
    (39, 40, 'r'),
    (10, 41, 'r'),
    (41, 42, 'r'),
    (42, 43, 'r'),
    (1, 44, 'r'),
    (44, 45, 'r'),
    (45, 47, 'r'),
    (44, 46, 'r'),
    (46, 48, 'r'),
]

## with simple fingers
pairs_simple = [
    (0, 1, 'r'),
    (1, 2, 'r'),
    (1, 3, 'r'),
    (2, 4, 'r'),
    (4, 6, 'r'),
    (6, 13, 'r'),
    (6, 14, 'r'),
    (6, 15, 'r'),
    (6, 16, 'r'),
    (6, 17, 'r'),

    (3, 5, 'r'),
    (5, 7, 'r'),
    (7, 18, 'r'),
    (7, 19, 'r'),
    (7, 20, 'r'),
    (7, 21, 'r'),
    (7, 22, 'r'),

    (1, 8, 'r'),
    (8, 9, 'r'),
    (8, 10, 'r'),
    (9, 11, 'r'),
    (10, 12, 'r'),
]
###############################################################################
def draw_3d_on_image(img, proj_joints, height, width, thickness=15):
    if proj_joints == []:
        return img

    new_img = img.copy()
    for pair in pairs_simple:
        pt1 = (int(proj_joints[pair[0]][0]), int(proj_joints[pair[0]][1]))
        pt2 = (int(proj_joints[pair[1]][0]), int(proj_joints[pair[1]][1]))
        if pt1[0] >= width or pt1[0] <= 0 or pt1[1] >= height or pt1[1] <= 0 or pt2[0] >= width or pt2[0] <= 0 or pt2[1] >= height or pt2[1] <= 0:
            pass
        else:
            rgb = [v * 255 for v in matplotlib.colors.to_rgba(pair[2])][:3]
            cv2.line(new_img, pt1, pt2, color=rgb[::-1], thickness=thickness)

    return new_img

# SKELETON
def draw_skeleton_on_image(img, skeleton, thickness=15):
    if not skeleton:
        return img

    new_img = img.copy()
    for pair in SkeletonWrapper.skeleton_line_pairs:
        pt1 = (int(skeleton[pair[0] * 3]), int(skeleton[pair[0] * 3 + 1]))
        pt2 = (int(skeleton[pair[1] * 3]), int(skeleton[pair[1] * 3 + 1]))
        if pt1[0] == 0 or pt2[1] == 0:
            pass
        else:
            rgb = [v * 255 for v in matplotlib.colors.to_rgba(pair[2])][:3]
            cv2.line(new_img, pt1, pt2, color=rgb[::-1], thickness=thickness)

    return new_img


def is_list_empty(my_list):
    return all(map(is_list_empty, my_list)) if isinstance(my_list, list) else False


def get_closest_skeleton(frame, selected_body):
    """ find the closest one to the selected skeleton """
    diff_idx = [i * 3 for i in range(8)] + [i * 3 + 1 for i in range(8)]  # upper-body

    min_diff = 10000000
    tracked_person = None
    for person in frame:  # people
        body = get_skeleton_from_frame(person)

        diff = 0
        n_diff = 0
        for i in diff_idx:
            if body[i] > 0 and selected_body[i] > 0:
                diff += abs(body[i] - selected_body[i])
                n_diff += 1
        if n_diff > 0:
            diff /= n_diff
        if diff < min_diff:
            min_diff = diff
            tracked_person = person

    base_distance = max(abs(selected_body[0 * 3 + 1] - selected_body[1 * 3 + 1]) * 3,
                        abs(selected_body[2 * 3] - selected_body[5 * 3]) * 2)
    if tracked_person and min_diff > base_distance:  # tracking failed
        tracked_person = None

    return tracked_person


def get_skeleton_from_frame(frame):
    if 'pose_keypoints_2d' in frame:
        return frame['pose_keypoints_2d']
    elif 'pose_keypoints' in frame:
        return frame['pose_keypoints']
    else:
        return None


class SkeletonWrapper:
    # color names: https://matplotlib.org/mpl_examples/color/named_colors.png
    visualization_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'gold'), (1, 5, 'darkgreen'), (5, 6, 'g'),
                                (6, 7, 'lightgreen'),
                                (1, 8, 'darkcyan'), (8, 9, 'c'), (9, 10, 'skyblue'), (1, 11, 'deeppink'), (11, 12, 'hotpink'), (12, 13, 'lightpink')]
    skeletons = []
    skeleton_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'gold'), (1, 5, 'darkgreen'),
                           (5, 6, 'g'), (6, 7, 'lightgreen')]

    def __init__(self, basepath, vid):
        # load skeleton data (and save it to pickle for next load)
        pickle_file = glob.glob(basepath + '/' + vid + '.pickle')

        if pickle_file:
            with open(pickle_file[0], 'rb') as file:
                self.skeletons = pickle.load(file)
        else:
            files = glob.glob(basepath + '/' + vid + '/*.json')
            if len(files) > 10:
                files = sorted(files)
                self.skeletons = []
                for file in files:
                    self.skeletons.append(self.read_skeleton_json(file))
                with open(basepath + '/' + vid + '.pickle', 'wb') as file:
                    pickle.dump(self.skeletons, file)
            else:
                self.skeletons = []


    def read_skeleton_json(self, file):
        with open(file) as json_file:
            skeleton_json = json.load(json_file)
            return skeleton_json['people']


    def get(self, start_frame_no, end_frame_no, interval=1):

        chunk = self.skeletons[start_frame_no:end_frame_no]

        if is_list_empty(chunk):
            return []
        else: 
            if interval > 1:
                return chunk[::int(interval)]
            else:
                return chunk


###############################################################################
# VIDEO
def read_video(base_path, vid):
    files = glob.glob(base_path + '/*' + vid + '.mp4')
    if len(files) == 0:
        return None
    elif len(files) >= 2:
        assert False
    filepath = files[0]

    video_obj = VideoWrapper(filepath)

    return video_obj


class VideoWrapper:
    video = []

    def __init__(self, filepath):
        self.filepath = filepath
        self.video = cv2.VideoCapture(filepath)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.framerate = self.video.get(cv2.CAP_PROP_FPS)

    def get_video_reader(self):
        return self.video

    def frame2second(self, frame_no):
        return frame_no / self.framerate

    def second2frame(self, second):
        return int(round(second * self.framerate))

    def set_current_frame(self, cur_frame_no):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_no)


###############################################################################
# CLIP
def load_clip_data(vid):
    try:
        with open("{}/{}.json".format(my_config.CLIP_PATH, vid)) as data_file:
            data = json.load(data_file)
            return data
    except FileNotFoundError:
        return None


def load_clip_filtering_aux_info(vid):
    try:
        with open("{}/{}_aux_info.json".format(my_config.CLIP_PATH, vid)) as data_file:
            data = json.load(data_file)
            return data
    except FileNotFoundError:
        return None


#################################################################################
#SUBTITLE
class SubtitleWrapper:
    TIMESTAMP_PATTERN = re.compile('(\d+)?:?(\d{2}):(\d{2})[.,](\d{3})')

    def __init__(self, vid, mode):
        self.subtitle = []
        if mode == 'auto':
            self.load_auto_subtitle_data(vid)
        elif mode == 'gentle':
            self.load_gentle_subtitle(vid)

    def get(self):
        return self.subtitle

    # using gentle lib
    def load_gentle_subtitle(self,vid):
        try:
            with open("{}/{}_align_results.json".format(my_config.GENTLE_PATH, vid)) as data_file:
                data = json.load(data_file)
                if 'words' in data:
                    raw_subtitle = data['words']

                    for word in raw_subtitle :
                        if word['case'] == 'success':
                            self.subtitle.append(word)
                else:
                    self.subtitle = None
                return data
        except FileNotFoundError:
            self.subtitle = None

    # using youtube automatic subtitle
    def load_auto_subtitle_data(self, vid):
        lang = my_config.LANG
        postfix_in_filename = '-'+lang+'-auto.vtt'
        file_list = glob.glob(my_config.SUBTITLE_PATH + '/*' + vid + postfix_in_filename)
        if len(file_list) > 1:
            print('more than one subtitle. check this.', file_list)
            self.subtitle = None
            assert False
        if len(file_list) == 1:
            for i, subtitle_chunk in enumerate(WebVTT().read(file_list[0])):
                raw_subtitle = str(subtitle_chunk.raw_text)
                if raw_subtitle.find('\n'):
                    raw_subtitle = raw_subtitle.split('\n')

                for raw_subtitle_chunk in raw_subtitle:
                    if self.TIMESTAMP_PATTERN.search(raw_subtitle_chunk) is None:
                        continue

                    # removes html tags and timing tags from caption text
                    raw_subtitle_chunk = raw_subtitle_chunk.replace("</c>", "")
                    raw_subtitle_chunk = re.sub("<c[.]\w+>", '', raw_subtitle_chunk)

                    word_list = []
                    raw_subtitle_s = subtitle_chunk.start_in_seconds
                    raw_subtitle_e = subtitle_chunk.end_in_seconds

                    word_chunk = raw_subtitle_chunk.split('<c>')

                    for i, word in enumerate(word_chunk):
                        word_info = {}

                        if i == len(word_chunk)-1:
                            word_info['word'] = word
                            word_info['start'] = word_list[i-1]['end']
                            word_info['end'] = raw_subtitle_e
                            word_list.append(word_info)
                            break

                        word = word.split("<")
                        word_info['word'] = word[0]
                        word_info['end'] = self.get_seconds(word[1][:-1])

                        if i == 0:
                            word_info['start'] = raw_subtitle_s
                            word_list.append(word_info)
                            continue

                        word_info['start'] = word_list[i-1]['end']
                        word_list.append(word_info)

                    self.subtitle.extend(word_list)
        else:
            print('subtitle file is not exist')
            self.subtitle = None

    # convert timestamp to second
    def get_seconds(self, word_time_e):
        time_value = re.match(self.TIMESTAMP_PATTERN, word_time_e)
        if not time_value:
            print('wrong time stamp pattern')
            exit()

        values = list(map(lambda x: int(x) if x else 0, time_value.groups()))
        hours, minutes, seconds, milliseconds = values[0], values[1], values[2], values[3]

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
