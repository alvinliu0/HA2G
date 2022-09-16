import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

angle_pair = [
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8)
]
avg_angle = [0.22037504613399506, 0.4590071439743042, 0.22463147342205048, 0.45562979578971863]
var_angle = [0.0018439559498801827, 0.013570506125688553, 0.0017794054001569748, 0.013684595935046673]

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
        cross_dist = 1.0 / (self.l2_sim(face_feat, audio_feat) + 1e-8)

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
        # print(self.l2_sim(face_feat, audio_feat))
        cross_dist = 1.0 / (self.l2_sim(face_feat, audio_feat) + 1e-8)
        cross_dist = torch.clamp(cross_dist, min=1e-8)

        if mode == 'max':
            label = torch.arange(face_feat.size(0)).to(cross_dist.device)
            loss = F.cross_entropy(cross_dist, label)
        else:
            raise ValueError
        return loss


def train_iter_hierarchy(args, epoch, in_text_padded, in_spec, target, vid_indices,
                   g1, g2, g3, discriminator, audio_encoder, text_encoder,
                   gen_optimizer_1, gen_optimizer_2, gen_optimizer_3, dis_optimizer, 
                   audio_optimizer, text_optimizer):
    warm_up_epochs = args.loss_warmup
    use_noisy_target = False

    weight, feat_low, feat_mid, feat_high, linear_blend_feat = audio_encoder(in_spec, vid_indices)
    text_feat = text_encoder(in_text_padded)

    # make pre seq input
    pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
    pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
    pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

    target_1 = torch.cat((target[:, :, :4 * 3], target[:, :, 6 * 3:7 * 3]), dim = 2)
    target_2 = torch.cat((target[:, :, :5 * 3], target[:, :, 6 * 3:8 * 3]), dim = 2)
    target_3 = target

    ###########################################################################################
    # train D
    dis_error = None
    if epoch > warm_up_epochs and args.loss_gan_weight > 0.0:
        dis_optimizer.zero_grad()

        # _, _, _, _, linear_blend_feat, z_context, _, _ = audio_encoder(in_spec, vid_indices)
        # _, _, _, _, linear_blend_feat = audio_encoder(in_spec, vid_indices)
        # text_feat, _ = text_encoder(in_text_padded)

        pre_seq_1 = target_1.new_zeros((target_1.shape[0], target_1.shape[1], target_1.shape[2] + 1))
        pre_seq_1[:, 0:args.n_pre_poses, :-1] = target_1[:, 0:args.n_pre_poses, :]
        pre_seq_1[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        out_dir_vec_1, *_ = g1(pre_seq_1, in_text_padded, linear_blend_feat[0], vid_indices)

        pre_seq_2 = target_2.new_zeros((target_2.shape[0], target_2.shape[1], target_2.shape[2] + 1))
        pre_seq_2[:, 0:args.n_pre_poses, :-1] = target_2[:, 0:args.n_pre_poses, :]
        pre_seq_2[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_2[:, args.n_pre_poses:, :4 * 3] = out_dir_vec_1[:, args.n_pre_poses:, :4 * 3]
        pre_seq_2[:, args.n_pre_poses:, 5 * 3:6 * 3] = out_dir_vec_1[:, args.n_pre_poses:, 4 * 3:5 * 3]
        out_dir_vec_2, *_ = g2(pre_seq_2, in_text_padded, linear_blend_feat[1], vid_indices)  # out shape (batch x seq x dim)

        pre_seq_3 = target_3.new_zeros((target_3.shape[0], target_3.shape[1], target_3.shape[2] + 1))
        pre_seq_3[:, 0:args.n_pre_poses, :-1] = target_3[:, 0:args.n_pre_poses, :]
        pre_seq_3[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_3[:, args.n_pre_poses:, :5 * 3] = out_dir_vec_2[:, args.n_pre_poses:, :5 * 3]
        pre_seq_3[:, args.n_pre_poses:, 6 * 3:8 * 3] = out_dir_vec_2[:, args.n_pre_poses:, 5 * 3:7 * 3]
        out_dir_vec, *_ = g3(pre_seq_3, in_text_padded, linear_blend_feat[2], vid_indices)  # out shape (batch x seq x dim)

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
    out_dir_vec_1, *_ = g1(pre_seq_1, in_text_padded, linear_blend_feat[0], vid_indices)

    pre_seq_2 = target_2.new_zeros((target_2.shape[0], target_2.shape[1], target_2.shape[2] + 1))
    pre_seq_2[:, 0:args.n_pre_poses, :-1] = target_2[:, 0:args.n_pre_poses, :]
    pre_seq_2[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
    pre_seq_2[:, args.n_pre_poses:, :4 * 3] = out_dir_vec_1[:, args.n_pre_poses:, :4 * 3]
    pre_seq_2[:, args.n_pre_poses:, 5 * 3:6 * 3] = out_dir_vec_1[:, args.n_pre_poses:, 4 * 3:5 * 3]
    out_dir_vec_2, *_ = g2(pre_seq_2, in_text_padded, linear_blend_feat[1], vid_indices)  # out shape (batch x seq x dim)

    pre_seq_3 = target_3.new_zeros((target_3.shape[0], target_3.shape[1], target_3.shape[2] + 1))
    pre_seq_3[:, 0:args.n_pre_poses, :-1] = target_3[:, 0:args.n_pre_poses, :]
    pre_seq_3[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
    pre_seq_3[:, args.n_pre_poses:, :5 * 3] = out_dir_vec_2[:, args.n_pre_poses:, :5 * 3]
    pre_seq_3[:, args.n_pre_poses:, 6 * 3:8 * 3] = out_dir_vec_2[:, args.n_pre_poses:, 5 * 3:7 * 3]
    out_dir_vec, z_context, z_mu, z_logvar = g3(pre_seq_3, in_text_padded, linear_blend_feat[2], vid_indices)  # out shape (batch x seq x dim)

    # loss
    beta = 0.1
    huber_loss = F.smooth_l1_loss(out_dir_vec_1 / beta, target_1 / beta) * beta + \
                    F.smooth_l1_loss(out_dir_vec_2 / beta, target_2 / beta) * beta + \
                        F.smooth_l1_loss(out_dir_vec / beta, target_3 / beta) * beta
    # huber_loss = F.smooth_l1_loss(out_dir_vec / beta, target_3 / beta) * beta
    final_loss = (F.smooth_l1_loss(out_dir_vec / beta, target_3 / beta) * beta).item()
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
        out_dir_vec_rand_1, *_ = g1(pre_seq_1, in_text_padded, linear_blend_feat[0], rand_vids)

        pre_seq_2 = target_2.new_zeros((target_2.shape[0], target_2.shape[1], target_2.shape[2] + 1))
        pre_seq_2[:, 0:args.n_pre_poses, :-1] = target_2[:, 0:args.n_pre_poses, :]
        pre_seq_2[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_2[:, args.n_pre_poses:, :4 * 3] = out_dir_vec_rand_1[:, args.n_pre_poses:, :4 * 3]
        pre_seq_2[:, args.n_pre_poses:, 5 * 3:6 * 3] = out_dir_vec_rand_1[:, args.n_pre_poses:, 4 * 3:5 * 3]
        out_dir_vec_rand_2, *_ = g2(pre_seq_2, in_text_padded, linear_blend_feat[1], rand_vids)  # out shape (batch x seq x dim)

        pre_seq_3 = target_3.new_zeros((target_3.shape[0], target_3.shape[1], target_3.shape[2] + 1))
        pre_seq_3[:, 0:args.n_pre_poses, :-1] = target_3[:, 0:args.n_pre_poses, :]
        pre_seq_3[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_3[:, args.n_pre_poses:, :5 * 3] = out_dir_vec_rand_2[:, args.n_pre_poses:, :5 * 3]
        pre_seq_3[:, args.n_pre_poses:, 6 * 3:8 * 3] = out_dir_vec_rand_2[:, args.n_pre_poses:, 5 * 3:7 * 3]
        out_dir_vec_rand_vid, z_context_rand, _, _ = g3(pre_seq_3, in_text_padded, linear_blend_feat[2], rand_vids)  # out shape (batch x seq x dim)

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

