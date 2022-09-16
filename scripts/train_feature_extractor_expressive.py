import pprint
import configargparse
import time
from pathlib import Path
import sys
from tqdm import tqdm

[sys.path.append(i) for i in ['.', '..']]

import matplotlib
import torch.nn as nn
import torch.nn.functional as F

from utils.vocab_utils import build_vocab

matplotlib.use('Agg')  # we don't use interactive GUI

from parse_args import parse_args

from torch import optim

from data_loader.lmdb_data_loader import *
import utils.train_utils

from model.motion_ae import *

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        utils.set_random_seed(args.random_seed)
    
    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

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
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                             collate_fn=collate_fn
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
    
    # init hyper-parameter
    pose_dim = args.pose_dim
    latent_dim = args.latent_dim

    motion_ae = MotionAE(pose_dim, latent_dim).to(device)
    if torch.cuda.device_count() > 1:
        motion_ae = nn.DataParallel(motion_ae, list(range(torch.cuda.device_count())))

    optimizer = optim.Adam(motion_ae.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    loss_fn = nn.MSELoss()

    best_loss = 1e8
    best_loss_mse = 1e8
    best_loss_cos = 1e8
    best_epoch = -1
    best_state_dict = {}

    for epoch in range(1, args.epochs + 1):
        loss_sum = 0
        loss_sum_cos = 0
        loss_sum_mse = 0
        loss_sum_test = 0
        loss_sum_test_cos = 0
        loss_sum_test_mse = 0

        motion_ae.train()

        for idx, data in enumerate(train_loader):
            motion_ae.zero_grad()
            _, _, _, _, target_vec, _, _, _ = data
            target_vec = target_vec.to(device)
            # target_vec = target_vec.view(target_vec.shape[0], target_vec.shape[1], -1, 3)

            pred, z = motion_ae(target_vec)

            reconstruction_loss = F.l1_loss(pred, target_vec, reduction='none')
            reconstruction_loss = torch.mean(reconstruction_loss, dim=(1, 2))

            if True:  # use pose diff
                target_diff = target_vec[:, 1:] - target_vec[:, :-1]
                recon_diff = pred[:, 1:] - pred[:, :-1]
                reconstruction_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

            reconstruction_loss = torch.sum(reconstruction_loss)
            
            cos_loss = torch.sum(1 - torch.cosine_similarity(pred.view(pred.shape[0], pred.shape[1], -1, 3), target_vec.view(target_vec.shape[0], target_vec.shape[1], -1, 3), dim = -1))
            
            loss = args.cos_loss_weight * cos_loss + reconstruction_loss
            loss_sum += args.cos_loss_weight * cos_loss.item() + reconstruction_loss.item()
            loss_sum_cos += cos_loss.item()
            loss_sum_mse += reconstruction_loss.item()
            
            loss.backward()
            optimizer.step()

        motion_ae.eval()
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                _, _, _, _, target_vec, _, _, _ = data
                target_vec = target_vec.to(device)
                # target_vec = target_vec.view(target_vec.shape[0], target_vec.shape[1], -1, 3)

                pred, z = motion_ae(target_vec)

                reconstruction_loss = F.l1_loss(pred, target_vec, reduction='none')
                reconstruction_loss = torch.mean(reconstruction_loss, dim=(1, 2))

                if True:  # use pose diff
                    target_diff = target_vec[:, 1:] - target_vec[:, :-1]
                    recon_diff = pred[:, 1:] - pred[:, :-1]
                    reconstruction_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

                reconstruction_loss = torch.sum(reconstruction_loss)
                
                cos_loss = torch.sum(1 - torch.cosine_similarity(pred.view(pred.shape[0], pred.shape[1], -1, 3), target_vec.view(target_vec.shape[0], target_vec.shape[1], -1, 3), dim = -1))
                
                loss_sum_test += args.cos_loss_weight * cos_loss.item() + reconstruction_loss.item()
                loss_sum_test_cos += cos_loss.item()
                loss_sum_test_mse += reconstruction_loss.item()
            
        avg_train_loss = loss_sum / (len(train_loader))
        avg_train_cos_loss = loss_sum_cos / (len(train_loader))
        avg_train_mse_loss = loss_sum_mse / (len(train_loader))
        # avg_train_kld_loss = loss_sum_kld / (len(train_loader))

        avg_val_loss = loss_sum_test / (len(val_loader))
        avg_val_cos_loss = loss_sum_test_cos / (len(val_loader))
        avg_val_mse_loss = loss_sum_test_mse / (len(val_loader))
        # avg_val_kld_loss = loss_sum_test_kld / (len(val_loader))

        print("###############################################################")
        print(f"Avg train loss of epoch {epoch}: {avg_train_loss}, cosine loss {avg_train_cos_loss}, per vec {avg_train_cos_loss / (34 * args.batch_size * 42)}, MSE loss {avg_train_mse_loss}")
        print(f"Avg test loss of epoch {epoch}: {avg_val_loss}, cosine loss {avg_val_cos_loss}, per vec is {avg_val_cos_loss / (34 * args.batch_size * 42)}, MSE loss {avg_val_mse_loss}")

        if (avg_val_loss < best_loss):
            best_loss = avg_val_loss
            best_loss_mse = avg_val_mse_loss
            best_loss_cos = avg_val_cos_loss
            best_epoch = epoch
            try:  # multi gpu
                best_state_dict = motion_ae.module.state_dict()
            except AttributeError:  # single gpu
                best_state_dict = motion_ae.state_dict()
        if(epoch > 0 and epoch % 10 == 0):
            try:  # multi gpu
                motionae_state_dict = motion_ae.module.state_dict()
            except AttributeError:  # single gpu
                motionae_state_dict = motion_ae.state_dict()
            save_name = '{}/checkpoint_{}.bin'.format(args.model_save_path, epoch)
            torch.save({
                    'args': args, 'epoch': epoch, 'pose_dim': pose_dim, 'latent_dim': latent_dim, 'motion_ae': motionae_state_dict
                }, save_name)
        scheduler.step()
    print("best epoch is: {}".format(best_epoch))
    print("best val loss is: {}, MSE is {}, cos is {}".format(best_loss, best_loss_mse, best_loss_cos))
    save_name = '{}/checkpoint_best.bin'.format(args.model_save_path)
    torch.save({
            'args': args, 'epoch': best_epoch, 'pose_dim': pose_dim, 'latent_dim': latent_dim, 'motion_ae': best_state_dict
        }, save_name)

if __name__ == '__main__':
    _args = parse_args()

    main({'args': _args})
