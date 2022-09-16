import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

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
avg_angle = [0.5969760417938232, 0.572796642780304, 0.348366379737854, 0.5536502599716187, 0.13027764856815338, 
    0.2801012694835663, 0.21510013937950134, 0.2457924336194992, 0.25812962651252747, 0.1696397364139557, 
    0.22138600051403046, 0.2232128530740738, 0.10013844072818756, 0.13465291261672974, 0.15643933415412903, 
    0.0757620558142662, 0.08111366629600525, 0.07266224175691605, 0.28242993354797363, 0.5088332295417786, 
    0.13428474962711334, 0.31135401129722595, 0.21646016836166382, 0.26498687267303467, 0.2691807448863983, 
    0.18528689444065094, 0.23011097311973572, 0.23511438071727753, 0.08650383353233337, 0.11938644200563431, 
    0.16712385416030884, 0.07711927592754364, 0.08256717771291733, 0.07396762818098068, 0.2504960894584656, 
    0.508758008480072, 0.4859846234321594, 0.30816879868507385, 0.2943730056285858, 0.572842538356781, 
    0.4471983015537262]
var_angle = [0.00028363385354168713, 0.00029294739942997694, 0.001516797230578959, 0.010948357172310352, 
    0.0025349585339426994, 0.009562775492668152, 0.008637933991849422, 0.008715483359992504, 0.012276478111743927, 
    0.005242602434009314, 0.008161756210029125, 0.007505195681005716, 0.002306767040863633, 0.0008198867435567081, 
    9.477637649979442e-05, 4.9160284106619656e-05, 5.3111481975065544e-05, 4.9043188482755795e-05, 
    0.0013721085852012038, 0.010581498965620995, 0.00196851696819067, 0.006986899301409721, 0.006110062822699547, 
    0.0074407304637134075, 0.010817521251738071, 0.005984380841255188, 0.006697201170027256, 0.00707469554618001, 
    0.0020931533072143793, 0.0006661304505541921, 9.530011448077857e-05, 4.7486370021943e-05, 5.157381747267209e-05, 
    4.733635432785377e-05, 0.00095974380383268, 0.00023575413797516376, 0.0002760167117230594, 2.6063793484354392e-05, 
    2.591621523606591e-05, 0.01612936705350876, 0.013571133837103844]

def add_noise(data):
    noise = torch.randn_like(data) * 0.1
    return data + noise


class SoftmaxContrastiveLoss(nn.Module):
    def __init__(self):
        super(SoftmaxContrastiveLoss, self).__init__()
        self.cross_ent = nn.CrossEntropyLoss()

    def l2_norm(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        return x_norm

    def l2_sim(self, feature1, feature2):
        Feature = feature1.expand(feature1.size(0), feature1.size(0), feature1.size(1)).transpose(0, 1)
        return torch.norm(Feature - feature2, p=2, dim=2)

    @torch.no_grad()
    def evaluate(self, face_feat, audio_feat, mode='max'):
        assert mode in 'max' or 'confusion', '{} must be in max or confusion'.format(mode)
        face_feat = self.l2_norm(face_feat)
        audio_feat = self.l2_norm(audio_feat)
        cross_dist = 1.0 / self.l2_sim(face_feat, audio_feat)

        # print(cross_dist)
        if mode == 'max':
            label = torch.arange(face_feat.size(0)).to(cross_dist.device)
            max_idx = torch.argmax(cross_dist, dim=1)
            # print(max_idx, label)
            acc = torch.sum(label == max_idx).float() / label.size(0)
        else:
            raise ValueError
        # print(acc)
        return acc, cross_dist

    def forward(self, face_feat, audio_feat, mode='max'):
        assert mode in 'max' or 'confusion', '{} must be in max or confusion'.format(mode)

        face_feat = self.l2_norm(face_feat)
        audio_feat = self.l2_norm(audio_feat)

        cross_dist = 1.0 / self.l2_sim(face_feat, audio_feat)

        if mode == 'max':
            label = torch.arange(face_feat.size(0)).to(cross_dist.device)
            loss = F.cross_entropy(cross_dist, label)
        else:
            raise ValueError
        return loss


def train_iter_hierarchy_expressive(args, epoch, in_text_padded, in_spec, target, vid_indices,
                   g1, g2, g3, g4, g5, g6, discriminator, audio_encoder, text_encoder,
                   gen_optimizer_1, gen_optimizer_2, gen_optimizer_3, 
                   gen_optimizer_4, gen_optimizer_5, gen_optimizer_6, dis_optimizer, 
                   audio_optimizer, text_optimizer):
    warm_up_epochs = args.loss_warmup
    use_noisy_target = False

    weight, feat_low, feat_mid, feat_high, linear_blend_feat = audio_encoder(in_spec, vid_indices)
    text_feat = text_encoder(in_text_padded)

    # make pre seq input
    pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
    pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
    pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

    target_1 = torch.cat((target[:, :, :3 * 3], target[:, :, -5 * 3:]), dim = 2)
    target_2 = torch.cat((target[:, :, :4 * 3], target[:, :, 20 * 3:21 * 3], target[:, :, -5 * 3:]), dim = 2)
    target_3 = torch.cat((target[:, :, :5 * 3], target[:, :, 20 * 3:22 * 3], target[:, :, -5 * 3:]), dim = 2)
    target_4 = torch.cat((target[:, :, :6 * 3], target[:, :, 8 * 3:9 * 3], target[:, :, 11 * 3:12 * 3], target[:, :, 14 * 3:15 * 3], target[:, :, 17 * 3:18 * 3], target[:, :, 20 * 3:23 * 3], target[:, :, 25 * 3:26 * 3], target[:, :, 28 * 3:29 * 3], target[:, :, 31 * 3:32 * 3], target[:, :, 34 * 3:35 * 3], target[:, :, -5 * 3:]), dim = 2)
    target_5 = torch.cat((target[:, :, :7 * 3], target[:, :, 8 * 3:10 * 3], target[:, :, 11 * 3:13 * 3], target[:, :, 14 * 3:16 * 3], target[:, :, 17 * 3:19 * 3], target[:, :, 20 * 3:24 * 3], target[:, :, 25 * 3:27 * 3], target[:, :, 28 * 3:30 * 3], target[:, :, 31 * 3:33 * 3], target[:, :, 34 * 3:36 * 3], target[:, :, -5 * 3:]), dim = 2)
    target_6 = target

    ###########################################################################################
    # train D
    dis_error = None
    if epoch > warm_up_epochs and args.loss_gan_weight > 0.0:
        dis_optimizer.zero_grad()

        # _, _, _, _, linear_blend_feat = audio_encoder(in_spec, vid_indices)

        pre_seq_1 = target_1.new_zeros((target_1.shape[0], target_1.shape[1], target_1.shape[2] + 1))
        pre_seq_1[:, 0:args.n_pre_poses, :-1] = target_1[:, 0:args.n_pre_poses, :]
        pre_seq_1[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        out_dir_vec_1, *_ = g1(pre_seq_1, in_text_padded, linear_blend_feat[0], vid_indices)  # out shape (batch x seq x dim)

        pre_seq_2 = target_2.new_zeros((target_2.shape[0], target_2.shape[1], target_2.shape[2] + 1))
        pre_seq_2[:, 0:args.n_pre_poses, :-1] = target_2[:, 0:args.n_pre_poses, :]
        pre_seq_2[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_2[:, args.n_pre_poses:, :3 * 3] = out_dir_vec_1[:, args.n_pre_poses:, :3 * 3]
        pre_seq_2[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_1[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_2, *_ = g2(pre_seq_2, in_text_padded, linear_blend_feat[1], vid_indices)  # out shape (batch x seq x dim)

        pre_seq_3 = target_3.new_zeros((target_3.shape[0], target_3.shape[1], target_3.shape[2] + 1))
        pre_seq_3[:, 0:args.n_pre_poses, :-1] = target_3[:, 0:args.n_pre_poses, :]
        pre_seq_3[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_3[:, args.n_pre_poses:, :4 * 3] = out_dir_vec_2[:, args.n_pre_poses:, :4 * 3]
        pre_seq_3[:, args.n_pre_poses:, 5 * 3:6 * 3] = out_dir_vec_2[:, args.n_pre_poses:, 4 * 3:5 * 3]
        pre_seq_3[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_2[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_3, *_ = g3(pre_seq_3, in_text_padded, linear_blend_feat[2], vid_indices)  # out shape (batch x seq x dim)

        pre_seq_4 = target_4.new_zeros((target_4.shape[0], target_4.shape[1], target_4.shape[2] + 1))
        pre_seq_4[:, 0:args.n_pre_poses, :-1] = target_4[:, 0:args.n_pre_poses, :]
        pre_seq_4[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_4[:, args.n_pre_poses:, :5 * 3] = out_dir_vec_3[:, args.n_pre_poses:, :5 * 3]
        pre_seq_4[:, args.n_pre_poses:, 10 * 3:12 * 3] = out_dir_vec_3[:, args.n_pre_poses:, 5 * 3:7 * 3]
        pre_seq_4[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_3[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_4, *_ = g4(pre_seq_4, in_text_padded, linear_blend_feat[3], vid_indices)  # out shape (batch x seq x dim)

        pre_seq_5 = target_5.new_zeros((target_5.shape[0], target_5.shape[1], target_5.shape[2] + 1))
        pre_seq_5[:, 0:args.n_pre_poses, :-1] = target_5[:, 0:args.n_pre_poses, :]
        pre_seq_5[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_5[:, args.n_pre_poses:, :6 * 3] = out_dir_vec_4[:, args.n_pre_poses:, :6 * 3]
        pre_seq_5[:, args.n_pre_poses:, 7 * 3:8 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 6 * 3:7 * 3]
        pre_seq_5[:, args.n_pre_poses:, 9 * 3:10 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 7 * 3:8 * 3]
        pre_seq_5[:, args.n_pre_poses:, 11 * 3:12 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 8 * 3:9 * 3]
        pre_seq_5[:, args.n_pre_poses:, 13 * 3:14 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 9 * 3:10 * 3]
        pre_seq_5[:, args.n_pre_poses:, 15 * 3:18 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 10 * 3:13 * 3]
        pre_seq_5[:, args.n_pre_poses:, 19 * 3:20 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 13 * 3:14 * 3]
        pre_seq_5[:, args.n_pre_poses:, 21 * 3:22 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 14 * 3:15 * 3]
        pre_seq_5[:, args.n_pre_poses:, 23 * 3:24 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 15 * 3:16 * 3]
        pre_seq_5[:, args.n_pre_poses:, 25 * 3:26 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 16 * 3:17 * 3]
        pre_seq_5[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_4[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_5, *_ = g5(pre_seq_5, in_text_padded, linear_blend_feat[4], vid_indices)  # out shape (batch x seq x dim)

        pre_seq_6 = target_6.new_zeros((target_6.shape[0], target_6.shape[1], target_6.shape[2] + 1))
        pre_seq_6[:, 0:args.n_pre_poses, :-1] = target_6[:, 0:args.n_pre_poses, :]
        pre_seq_6[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_6[:, args.n_pre_poses:, :7 * 3] = out_dir_vec_5[:, args.n_pre_poses:, :7 * 3]
        pre_seq_6[:, args.n_pre_poses:, 8 * 3:10 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 7 * 3:9 * 3]
        pre_seq_6[:, args.n_pre_poses:, 11 * 3:13 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 9 * 3:11 * 3]
        pre_seq_6[:, args.n_pre_poses:, 14 * 3:16 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 11 * 3:13 * 3]
        pre_seq_6[:, args.n_pre_poses:, 17 * 3:19 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 13 * 3:15 * 3]
        pre_seq_6[:, args.n_pre_poses:, 20 * 3:24 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 15 * 3:19 * 3]
        pre_seq_6[:, args.n_pre_poses:, 25 * 3:27 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 19 * 3:21 * 3]
        pre_seq_6[:, args.n_pre_poses:, 28 * 3:30 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 21 * 3:23 * 3]
        pre_seq_6[:, args.n_pre_poses:, 31 * 3:33 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 23 * 3:25 * 3]
        pre_seq_6[:, args.n_pre_poses:, 34 * 3:36 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 25 * 3:27 * 3]
        pre_seq_6[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_5[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec, *_ = g6(pre_seq_6, in_text_padded, linear_blend_feat[5], vid_indices)  # out shape (batch x seq x dim)

        if use_noisy_target:
            noise_target = add_noise(target)
            noise_out = add_noise(out_dir_vec.detach())
            dis_real = discriminator(noise_target, in_text_padded)
            dis_fake = discriminator(noise_out, in_text_padded)
        else:
            dis_real = discriminator(target, in_text_padded)
            dis_fake = discriminator(out_dir_vec.detach(), in_text_padded)

        dis_error = torch.sum(-torch.mean(torch.log(dis_real + 1e-8) + torch.log(1 - dis_fake + 1e-8)))  # ns-gan
        dis_error.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=discriminator.parameters(), max_norm=1, norm_type=2)
        dis_optimizer.step()

    ###########################################################################################
    # train G
    gen_optimizer_1.zero_grad()
    gen_optimizer_2.zero_grad()
    gen_optimizer_3.zero_grad()
    gen_optimizer_4.zero_grad()
    gen_optimizer_5.zero_grad()
    gen_optimizer_6.zero_grad()
    audio_optimizer.zero_grad()
    text_optimizer.zero_grad()

    # weight, feat_low, feat_mid, feat_high, linear_blend_feat, z_context, z_mu, z_logvar = audio_encoder(in_spec, vid_indices)
    # weight, feat_low, feat_mid, feat_high, linear_blend_feat = audio_encoder(in_spec, vid_indices)
    # text_feat, _ = text_encoder(in_text_padded)

    criterion = SoftmaxContrastiveLoss()
    if args.loss_contrastive_pos_weight > 0.0:
        text_high_contrastive = criterion(text_feat.reshape(-1, text_feat.shape[2]), feat_high.reshape(-1, feat_high.shape[2]))
        # text_mid_contrastive = -criterion(text_feat.reshape(-1, text_feat.shape[2]), feat_mid.reshape(-1, feat_mid.shape[2]))
    if args.loss_contrastive_neg_weight > 0.0:
        text_low_contrastive = -criterion(text_feat.reshape(-1, text_feat.shape[2]), feat_low.reshape(-1, feat_low.shape[2]))

    # decoding
    pre_seq_1 = target_1.new_zeros((target_1.shape[0], target_1.shape[1], target_1.shape[2] + 1))
    pre_seq_1[:, 0:args.n_pre_poses, :-1] = target_1[:, 0:args.n_pre_poses, :]
    pre_seq_1[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
    out_dir_vec_1, *_ = g1(pre_seq_1, in_text_padded, linear_blend_feat[0], vid_indices)  # out shape (batch x seq x dim)

    pre_seq_2 = target_2.new_zeros((target_2.shape[0], target_2.shape[1], target_2.shape[2] + 1))
    pre_seq_2[:, 0:args.n_pre_poses, :-1] = target_2[:, 0:args.n_pre_poses, :]
    pre_seq_2[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
    pre_seq_2[:, args.n_pre_poses:, :3 * 3] = out_dir_vec_1[:, args.n_pre_poses:, :3 * 3]
    pre_seq_2[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_1[:, args.n_pre_poses:, -5 * 3:]
    out_dir_vec_2, *_ = g2(pre_seq_2, in_text_padded, linear_blend_feat[1], vid_indices)  # out shape (batch x seq x dim)

    pre_seq_3 = target_3.new_zeros((target_3.shape[0], target_3.shape[1], target_3.shape[2] + 1))
    pre_seq_3[:, 0:args.n_pre_poses, :-1] = target_3[:, 0:args.n_pre_poses, :]
    pre_seq_3[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
    pre_seq_3[:, args.n_pre_poses:, :4 * 3] = out_dir_vec_2[:, args.n_pre_poses:, :4 * 3]
    pre_seq_3[:, args.n_pre_poses:, 5 * 3:6 * 3] = out_dir_vec_2[:, args.n_pre_poses:, 4 * 3:5 * 3]
    pre_seq_3[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_2[:, args.n_pre_poses:, -5 * 3:]
    out_dir_vec_3, *_ = g3(pre_seq_3, in_text_padded, linear_blend_feat[2], vid_indices)  # out shape (batch x seq x dim)

    pre_seq_4 = target_4.new_zeros((target_4.shape[0], target_4.shape[1], target_4.shape[2] + 1))
    pre_seq_4[:, 0:args.n_pre_poses, :-1] = target_4[:, 0:args.n_pre_poses, :]
    pre_seq_4[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
    pre_seq_4[:, args.n_pre_poses:, :5 * 3] = out_dir_vec_3[:, args.n_pre_poses:, :5 * 3]
    pre_seq_4[:, args.n_pre_poses:, 10 * 3:12 * 3] = out_dir_vec_3[:, args.n_pre_poses:, 5 * 3:7 * 3]
    pre_seq_4[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_3[:, args.n_pre_poses:, -5 * 3:]
    out_dir_vec_4, *_ = g4(pre_seq_4, in_text_padded, linear_blend_feat[3], vid_indices)  # out shape (batch x seq x dim)

    pre_seq_5 = target_5.new_zeros((target_5.shape[0], target_5.shape[1], target_5.shape[2] + 1))
    pre_seq_5[:, 0:args.n_pre_poses, :-1] = target_5[:, 0:args.n_pre_poses, :]
    pre_seq_5[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
    pre_seq_5[:, args.n_pre_poses:, :6 * 3] = out_dir_vec_4[:, args.n_pre_poses:, :6 * 3]
    pre_seq_5[:, args.n_pre_poses:, 7 * 3:8 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 6 * 3:7 * 3]
    pre_seq_5[:, args.n_pre_poses:, 9 * 3:10 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 7 * 3:8 * 3]
    pre_seq_5[:, args.n_pre_poses:, 11 * 3:12 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 8 * 3:9 * 3]
    pre_seq_5[:, args.n_pre_poses:, 13 * 3:14 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 9 * 3:10 * 3]
    pre_seq_5[:, args.n_pre_poses:, 15 * 3:18 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 10 * 3:13 * 3]
    pre_seq_5[:, args.n_pre_poses:, 19 * 3:20 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 13 * 3:14 * 3]
    pre_seq_5[:, args.n_pre_poses:, 21 * 3:22 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 14 * 3:15 * 3]
    pre_seq_5[:, args.n_pre_poses:, 23 * 3:24 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 15 * 3:16 * 3]
    pre_seq_5[:, args.n_pre_poses:, 25 * 3:26 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 16 * 3:17 * 3]
    pre_seq_5[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_4[:, args.n_pre_poses:, -5 * 3:]
    out_dir_vec_5, *_ = g5(pre_seq_5, in_text_padded, linear_blend_feat[4], vid_indices)  # out shape (batch x seq x dim)

    pre_seq_6 = target_6.new_zeros((target_6.shape[0], target_6.shape[1], target_6.shape[2] + 1))
    pre_seq_6[:, 0:args.n_pre_poses, :-1] = target_6[:, 0:args.n_pre_poses, :]
    pre_seq_6[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
    pre_seq_6[:, args.n_pre_poses:, :7 * 3] = out_dir_vec_5[:, args.n_pre_poses:, :7 * 3]
    pre_seq_6[:, args.n_pre_poses:, 8 * 3:10 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 7 * 3:9 * 3]
    pre_seq_6[:, args.n_pre_poses:, 11 * 3:13 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 9 * 3:11 * 3]
    pre_seq_6[:, args.n_pre_poses:, 14 * 3:16 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 11 * 3:13 * 3]
    pre_seq_6[:, args.n_pre_poses:, 17 * 3:19 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 13 * 3:15 * 3]
    pre_seq_6[:, args.n_pre_poses:, 20 * 3:24 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 15 * 3:19 * 3]
    pre_seq_6[:, args.n_pre_poses:, 25 * 3:27 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 19 * 3:21 * 3]
    pre_seq_6[:, args.n_pre_poses:, 28 * 3:30 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 21 * 3:23 * 3]
    pre_seq_6[:, args.n_pre_poses:, 31 * 3:33 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 23 * 3:25 * 3]
    pre_seq_6[:, args.n_pre_poses:, 34 * 3:36 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 25 * 3:27 * 3]
    pre_seq_6[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_5[:, args.n_pre_poses:, -5 * 3:]
    out_dir_vec, z_context, z_mu, z_logvar = g6(pre_seq_6, in_text_padded, linear_blend_feat[5], vid_indices)  # out shape (batch x seq x dim)

    # loss
    beta = 0.1
    huber_loss = F.smooth_l1_loss(out_dir_vec_1 / beta, target_1 / beta) * beta + \
                    F.smooth_l1_loss(out_dir_vec_2 / beta, target_2 / beta) * beta + \
                        F.smooth_l1_loss(out_dir_vec_3 / beta, target_3 / beta) * beta + \
                            F.smooth_l1_loss(out_dir_vec_4 / beta, target_4 / beta) * beta + \
                                F.smooth_l1_loss(out_dir_vec_5 / beta, target_5 / beta) * beta + \
                                    F.smooth_l1_loss(out_dir_vec / beta, target_6 / beta) * beta
    final_loss = (F.smooth_l1_loss(out_dir_vec / beta, target_6 / beta) * beta).item()
    dis_output = discriminator(out_dir_vec, in_text_padded)
    gen_error = -torch.mean(torch.log(dis_output + 1e-8))
    kld = div_reg = None

    if (args.z_type == 'speaker' or args.z_type == 'random') and args.loss_reg_weight > 0.0:
        if args.z_type == 'speaker':
            # enforcing divergent gestures btw original vid and other vid
            rand_idx = torch.randperm(vid_indices.shape[0])
            rand_vids = vid_indices[rand_idx]
        else:
            rand_vids = None

        # _, _, _, _, linear_blend_feat_rand, z_context_rand, _, _ = audio_encoder(in_spec, rand_vids)
        # _, _, _, _, linear_blend_feat_rand = audio_encoder(in_spec, rand_vids)

        pre_seq_1 = target_1.new_zeros((target_1.shape[0], target_1.shape[1], target_1.shape[2] + 1))
        pre_seq_1[:, 0:args.n_pre_poses, :-1] = target_1[:, 0:args.n_pre_poses, :]
        pre_seq_1[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        out_dir_vec_rand_vid_1, *_ = g1(pre_seq_1, in_text_padded, linear_blend_feat[0], rand_vids)  # out shape (batch x seq x dim)

        pre_seq_2 = target_2.new_zeros((target_2.shape[0], target_2.shape[1], target_2.shape[2] + 1))
        pre_seq_2[:, 0:args.n_pre_poses, :-1] = target_2[:, 0:args.n_pre_poses, :]
        pre_seq_2[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_2[:, args.n_pre_poses:, :3 * 3] = out_dir_vec_rand_vid_1[:, args.n_pre_poses:, :3 * 3]
        pre_seq_2[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_rand_vid_1[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_rand_vid_2, *_ = g2(pre_seq_2, in_text_padded, linear_blend_feat[1], rand_vids)  # out shape (batch x seq x dim)

        pre_seq_3 = target_3.new_zeros((target_3.shape[0], target_3.shape[1], target_3.shape[2] + 1))
        pre_seq_3[:, 0:args.n_pre_poses, :-1] = target_3[:, 0:args.n_pre_poses, :]
        pre_seq_3[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_3[:, args.n_pre_poses:, :4 * 3] = out_dir_vec_rand_vid_2[:, args.n_pre_poses:, :4 * 3]
        pre_seq_3[:, args.n_pre_poses:, 5 * 3:6 * 3] = out_dir_vec_rand_vid_2[:, args.n_pre_poses:, 4 * 3:5 * 3]
        pre_seq_3[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_rand_vid_2[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_rand_vid_3, *_ = g3(pre_seq_3, in_text_padded, linear_blend_feat[2], rand_vids)  # out shape (batch x seq x dim)

        pre_seq_4 = target_4.new_zeros((target_4.shape[0], target_4.shape[1], target_4.shape[2] + 1))
        pre_seq_4[:, 0:args.n_pre_poses, :-1] = target_4[:, 0:args.n_pre_poses, :]
        pre_seq_4[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_4[:, args.n_pre_poses:, :5 * 3] = out_dir_vec_rand_vid_3[:, args.n_pre_poses:, :5 * 3]
        pre_seq_4[:, args.n_pre_poses:, 10 * 3:12 * 3] = out_dir_vec_rand_vid_3[:, args.n_pre_poses:, 5 * 3:7 * 3]
        pre_seq_4[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_rand_vid_3[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_rand_vid_4, *_ = g4(pre_seq_4, in_text_padded, linear_blend_feat[3], rand_vids)  # out shape (batch x seq x dim)

        pre_seq_5 = target_5.new_zeros((target_5.shape[0], target_5.shape[1], target_5.shape[2] + 1))
        pre_seq_5[:, 0:args.n_pre_poses, :-1] = target_5[:, 0:args.n_pre_poses, :]
        pre_seq_5[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_5[:, args.n_pre_poses:, :6 * 3] = out_dir_vec_rand_vid_4[:, args.n_pre_poses:, :6 * 3]
        pre_seq_5[:, args.n_pre_poses:, 7 * 3:8 * 3] = out_dir_vec_rand_vid_4[:, args.n_pre_poses:, 6 * 3:7 * 3]
        pre_seq_5[:, args.n_pre_poses:, 9 * 3:10 * 3] = out_dir_vec_rand_vid_4[:, args.n_pre_poses:, 7 * 3:8 * 3]
        pre_seq_5[:, args.n_pre_poses:, 11 * 3:12 * 3] = out_dir_vec_rand_vid_4[:, args.n_pre_poses:, 8 * 3:9 * 3]
        pre_seq_5[:, args.n_pre_poses:, 13 * 3:14 * 3] = out_dir_vec_rand_vid_4[:, args.n_pre_poses:, 9 * 3:10 * 3]
        pre_seq_5[:, args.n_pre_poses:, 15 * 3:18 * 3] = out_dir_vec_rand_vid_4[:, args.n_pre_poses:, 10 * 3:13 * 3]
        pre_seq_5[:, args.n_pre_poses:, 19 * 3:20 * 3] = out_dir_vec_rand_vid_4[:, args.n_pre_poses:, 13 * 3:14 * 3]
        pre_seq_5[:, args.n_pre_poses:, 21 * 3:22 * 3] = out_dir_vec_rand_vid_4[:, args.n_pre_poses:, 14 * 3:15 * 3]
        pre_seq_5[:, args.n_pre_poses:, 23 * 3:24 * 3] = out_dir_vec_rand_vid_4[:, args.n_pre_poses:, 15 * 3:16 * 3]
        pre_seq_5[:, args.n_pre_poses:, 25 * 3:26 * 3] = out_dir_vec_rand_vid_4[:, args.n_pre_poses:, 16 * 3:17 * 3]
        pre_seq_5[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_rand_vid_4[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_rand_vid_5, *_ = g5(pre_seq_5, in_text_padded, linear_blend_feat[4], rand_vids)  # out shape (batch x seq x dim)

        pre_seq_6 = target_6.new_zeros((target_6.shape[0], target_6.shape[1], target_6.shape[2] + 1))
        pre_seq_6[:, 0:args.n_pre_poses, :-1] = target_6[:, 0:args.n_pre_poses, :]
        pre_seq_6[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_6[:, args.n_pre_poses:, :7 * 3] = out_dir_vec_rand_vid_5[:, args.n_pre_poses:, :7 * 3]
        pre_seq_6[:, args.n_pre_poses:, 8 * 3:10 * 3] = out_dir_vec_rand_vid_5[:, args.n_pre_poses:, 7 * 3:9 * 3]
        pre_seq_6[:, args.n_pre_poses:, 11 * 3:13 * 3] = out_dir_vec_rand_vid_5[:, args.n_pre_poses:, 9 * 3:11 * 3]
        pre_seq_6[:, args.n_pre_poses:, 14 * 3:16 * 3] = out_dir_vec_rand_vid_5[:, args.n_pre_poses:, 11 * 3:13 * 3]
        pre_seq_6[:, args.n_pre_poses:, 17 * 3:19 * 3] = out_dir_vec_rand_vid_5[:, args.n_pre_poses:, 13 * 3:15 * 3]
        pre_seq_6[:, args.n_pre_poses:, 20 * 3:24 * 3] = out_dir_vec_rand_vid_5[:, args.n_pre_poses:, 15 * 3:19 * 3]
        pre_seq_6[:, args.n_pre_poses:, 25 * 3:27 * 3] = out_dir_vec_rand_vid_5[:, args.n_pre_poses:, 19 * 3:21 * 3]
        pre_seq_6[:, args.n_pre_poses:, 28 * 3:30 * 3] = out_dir_vec_rand_vid_5[:, args.n_pre_poses:, 21 * 3:23 * 3]
        pre_seq_6[:, args.n_pre_poses:, 31 * 3:33 * 3] = out_dir_vec_rand_vid_5[:, args.n_pre_poses:, 23 * 3:25 * 3]
        pre_seq_6[:, args.n_pre_poses:, 34 * 3:36 * 3] = out_dir_vec_rand_vid_5[:, args.n_pre_poses:, 25 * 3:27 * 3]
        pre_seq_6[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_rand_vid_5[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_rand_vid, z_context_rand, _, _ = g6(pre_seq_6, in_text_padded, linear_blend_feat[5], rand_vids)  # out shape (batch x seq x dim)

        beta = 0.05
        pose_l1 = F.smooth_l1_loss(out_dir_vec / beta, out_dir_vec_rand_vid.detach() / beta, reduction='none') * beta
        pose_l1 = pose_l1.sum(dim=1).sum(dim=1)

        pose_l1 = pose_l1.view(pose_l1.shape[0], -1).mean(1)
        z_l1 = F.l1_loss(z_context.detach(), z_context_rand.detach(), reduction='none')
        z_l1 = z_l1.view(z_l1.shape[0], -1).mean(1)

        div_reg = -(pose_l1 / (z_l1 + 1.0e-5))
        div_reg = torch.clamp(div_reg, min=-1000)
        div_reg = div_reg.mean()

        if args.z_type == 'speaker':
            # speaker embedding KLD
            kld = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
            loss = args.loss_regression_weight * huber_loss + args.loss_kld_weight * kld + args.loss_reg_weight * div_reg
        else:
            loss = args.loss_regression_weight * huber_loss + args.loss_reg_weight * div_reg
    else:
        loss = args.loss_regression_weight * huber_loss #+ var_loss

    if epoch > warm_up_epochs:
        loss += args.loss_gan_weight * gen_error

    if args.loss_contrastive_pos_weight > 0.0:
        loss += args.loss_contrastive_pos_weight * text_high_contrastive
    if args.loss_contrastive_neg_weight > 0.0:
        loss += args.loss_contrastive_neg_weight * text_low_contrastive

    # physical constraint
    if args.loss_physical_weight > 0.0:
        batch_size = out_dir_vec.shape[0]
        physical_loss = 0
        raw_dir_vec = out_dir_vec + torch.tensor(args.mean_dir_vec).squeeze(1).unsqueeze(0).unsqueeze(0).to(out_dir_vec.device)
        left_palm = torch.cross(raw_dir_vec[:, :, 11 * 3 : 12 * 3], raw_dir_vec[:, :, 17 * 3 : 18 * 3], dim = 2)
        right_palm = torch.cross(raw_dir_vec[:, :, 28 * 3 : 29 * 3], raw_dir_vec[:, :, 34 * 3 : 35 * 3], dim = 2)
        raw_dir_vec = torch.cat((raw_dir_vec, left_palm, right_palm), dim = 2)
        raw_dir_vec = raw_dir_vec.reshape(raw_dir_vec.shape[0], raw_dir_vec.shape[1], -1, 3)
        raw_dir_vec = F.normalize(raw_dir_vec, dim = -1)

        all_vec = raw_dir_vec.reshape(raw_dir_vec.shape[0] * raw_dir_vec.shape[1], -1, 3)

        for idx, pair in enumerate(angle_pair):
            vec1 = all_vec[:, pair[0]]
            vec2 = all_vec[:, pair[1]]
            inner_product = torch.einsum('ij,ij->i', [vec1, vec2])
            inner_product = torch.clamp(inner_product, -1 + 1e-7, 1 - 1e-7, out=None)
            angle = torch.acos(inner_product) / math.pi
            # physical_loss += torch.mean(torch.abs(angle - avg_angle[idx]))
            # prob_loss = -torch.mean(torch.log(1e-8 + 1.0 / math.sqrt(2 * math.pi * var_angle[idx]) * torch.exp(-((angle - avg_angle[idx]) ** 2) / (2 * var_angle[idx]))))
            prob_loss = torch.mean(((angle - avg_angle[idx]) ** 2) / (2 * var_angle[idx]))
            physical_loss += prob_loss

        loss += args.loss_physical_weight * physical_loss

    loss.backward()
    # torch.nn.utils.clip_grad_norm_(parameters=g1.parameters(), max_norm=1, norm_type=2)
    # torch.nn.utils.clip_grad_norm_(parameters=g2.parameters(), max_norm=1, norm_type=2)
    # torch.nn.utils.clip_grad_norm_(parameters=g3.parameters(), max_norm=1, norm_type=2)
    # torch.nn.utils.clip_grad_norm_(parameters=audio_encoder.parameters(), max_norm=1, norm_type=2)
    # torch.nn.utils.clip_grad_norm_(parameters=text_encoder.parameters(), max_norm=1, norm_type=2)
    gen_optimizer_1.step()
    gen_optimizer_2.step()
    gen_optimizer_3.step()
    gen_optimizer_4.step()
    gen_optimizer_5.step()
    gen_optimizer_6.step()
    audio_optimizer.step()
    text_optimizer.step()

    ret_dict = {'loss': args.loss_regression_weight * huber_loss.item()}
    if kld:
        ret_dict['KLD'] = args.loss_kld_weight * kld.item()
    if div_reg:
        ret_dict['DIV_REG'] = args.loss_reg_weight * div_reg.item()

    if epoch > warm_up_epochs and args.loss_gan_weight > 0.0:
        ret_dict['gen'] = args.loss_gan_weight * gen_error.item()
        ret_dict['dis'] = dis_error.item()

    if args.loss_contrastive_pos_weight > 0.0:
        ret_dict['c_pos'] = args.loss_contrastive_pos_weight * text_high_contrastive.item()
    if args.loss_contrastive_neg_weight > 0.0:
        ret_dict['c_neg'] = args.loss_contrastive_neg_weight * text_low_contrastive.item()
    if args.loss_physical_weight > 0.0:
        ret_dict['phy'] = args.loss_physical_weight * physical_loss.item()

    return ret_dict

