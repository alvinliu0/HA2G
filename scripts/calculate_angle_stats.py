import pprint
import time
from pathlib import Path
import sys

[sys.path.append(i) for i in ['.', '..']]

import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from model import speech2gesture, vocab
from utils.average_meter import AverageMeter
from utils.data_utils import convert_dir_vec_to_pose
from utils.vocab_utils import build_vocab

matplotlib.use('Agg')  # we don't use interactive GUI

from parse_args import parse_args
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator

import math

from data_loader.lmdb_data_loader import *
import utils.train_utils

import librosa
from librosa.feature import melspectrogram

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    collate_fn = default_collate_fn

    # dataset
    mean_dir_vec = np.array(args.mean_dir_vec).reshape(-1, 3)
    train_dataset = SpeechMotionDataset(args.train_data_path[0],
                                        n_poses=args.n_poses,
                                        subdivision_stride=args.subdivision_stride,
                                        pose_resampling_fps=args.motion_resampling_framerate,
                                        mean_dir_vec=mean_dir_vec,
                                        mean_pose=args.mean_pose,
                                        remove_word_timing=(args.input_context == 'text')
                                        )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )

    val_dataset = SpeechMotionDataset(args.val_data_path[0],
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      speaker_model=train_dataset.speaker_model,
                                      mean_dir_vec=mean_dir_vec,
                                      mean_pose=args.mean_pose,
                                      remove_word_timing=(args.input_context == 'text')
                                      )

    test_dataset = SpeechMotionDataset(args.test_data_path[0],
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate,
                                       speaker_model=train_dataset.speaker_model,
                                       mean_dir_vec=mean_dir_vec,
                                       mean_pose=args.mean_pose)

    # build vocab
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset, test_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    if args.pose_dim == 27:
        angle_pair = [
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8)
        ]
    elif args.pose_dim == 126:
        angle_pair = [
            (0, 1),
            (0, 2),
            (1, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (14, 15),
            (15, 16),
            (17, 18),
            (18, 19),
            (17, 5),
            (5, 8),
            (8, 14),
            (14, 11),
            (2, 20),
            (20, 21),
            (22, 23),
            (23, 24),
            (25, 26),
            (26, 27),
            (28, 29),
            (29, 30),
            (31, 32),
            (32, 33),
            (34, 35),
            (35, 36),
            (34, 22),
            (22, 25),
            (25, 31),
            (31, 28),
            (0, 37),
            (37, 38),
            (37, 39),
            (38, 40),
            (39, 41),
            # palm
            (4, 42),
            (21, 43)
        ]
    else:
        assert False
        
    avg_angle = [0] * len(angle_pair)
    var_angle = [0] * len(angle_pair)
    change_angle = [0] * len(angle_pair)

    cnt_angle = 0
    cnt_change = 0

    # stat angle
    for data in tqdm(train_loader):

        in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
        batch_size = target_vec.size(0)
        target_vec = target_vec + torch.tensor(args.mean_dir_vec).squeeze(1).unsqueeze(0).unsqueeze(0)
        target_vec = target_vec.to(device)
        if args.pose_dim == 126:
            left_palm = torch.cross(target_vec[:, :, 11 * 3 : 12 * 3], target_vec[:, :, 17 * 3 : 18 * 3], dim = 2)
            right_palm = torch.cross(target_vec[:, :, 28 * 3 : 29 * 3], target_vec[:, :, 34 * 3 : 35 * 3], dim = 2)
            target_vec = torch.cat((target_vec, left_palm, right_palm), dim = 2)
        target_vec = target_vec.reshape(target_vec.shape[0], target_vec.shape[1], -1, 3)
        target_vec = F.normalize(target_vec, dim = -1)

        angle_batch = target_vec.shape[0] * target_vec.shape[1]
        change_batch = target_vec.shape[0] * (target_vec.shape[1] - 1)
        all_vec = target_vec.reshape(target_vec.shape[0] * target_vec.shape[1], -1, 3)

        for idx, pair in enumerate(angle_pair):
            vec1 = all_vec[:, pair[0]]
            vec2 = all_vec[:, pair[1]]
            inner_product = torch.einsum('ij,ij->i', [vec1, vec2])
            inner_product = torch.clamp(inner_product, -1, 1, out=None)
            angle = torch.acos(inner_product) / math.pi
            angle_time = angle.reshape(batch_size, -1)
            angle_diff = torch.mean(torch.abs(angle_time[:, 1:] - angle_time[:, :-1]))
            avg_batch = torch.mean(angle)
            var_batch = torch.var(angle)
            if (torch.isnan(angle_diff)):
                angle_diff = change_angle[idx]
            if (torch.isnan(avg_batch)):
                avg_batch = avg_angle[idx]
            if (torch.isnan(var_batch)):
                var_batch = var_angle[idx]
            history_avg = avg_angle[idx]
            change_angle[idx] = (cnt_change * change_angle[idx] + angle_diff * change_batch) / (cnt_change + change_batch)
            avg_angle[idx] = (cnt_angle * avg_angle[idx] + angle_batch * avg_batch) / (cnt_angle + angle_batch)
            var_angle[idx] = (cnt_angle * (var_angle[idx] + torch.pow((avg_angle[idx] - history_avg), 2)) + angle_batch * (var_batch + torch.pow((avg_angle[idx] - avg_batch), 2))) / (cnt_angle + angle_batch)

        cnt_angle += angle_batch
        cnt_change += change_batch
    change_angle = [x.item() for x in change_angle]
    avg_angle = [x.item() for x in avg_angle]
    var_angle = [x.item() for x in var_angle]

    print('change angle: ', change_angle)
    print('avg angle: ', avg_angle)
    print('var angle: ', var_angle)

if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
