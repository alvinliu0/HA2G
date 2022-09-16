import pprint
import time
from pathlib import Path
import sys

[sys.path.append(i) for i in ['.', '..']]

import matplotlib
import math
import librosa
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import speech2gesture, vocab
from model.embedding_net import EmbeddingNet
from model.seq2seq_net import Seq2SeqNet
from train_eval.train_gan import train_iter_gan
from train_eval.train_hierarchy_expressive import train_iter_hierarchy_expressive
from train_eval.train_joint_embed import train_iter_embed, eval_embed
from train_eval.train_seq2seq import train_iter_seq2seq
from train_eval.train_speech2gesture import train_iter_speech2gesture
from utils.average_meter import AverageMeter
from utils.data_utils import convert_dir_vec_to_pose
from utils.vocab_utils import build_vocab

matplotlib.use('Agg')  # we don't use interactive GUI

from parse_args import parse_args
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from model.multimodal_context_net import PoseGenerator, ConvDiscriminator
from model.hierarchy_net import Hierarchical_PoseGenerator, Hierarchical_ConvDiscriminator, Hierarchical_WavEncoder, TextEncoderTCN

from torch import optim

from data_loader.lmdb_data_loader_expressive import *
import utils.train_utils_expressive

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

change_angle = [0.0027804733254015446, 0.002761547453701496, 0.005953566171228886, 0.013764726929366589, 
    0.022748252376914024, 0.039307352155447006, 0.03733552247285843, 0.03775784373283386, 0.0485558956861496, 
    0.032914578914642334, 0.03800227493047714, 0.03757007420063019, 0.027338404208421707, 0.01640886254608631, 
    0.003166505601257086, 0.0017252820543944836, 0.0018696568440645933, 0.0016072227153927088, 0.005681346170604229, 
    0.013287615962326527, 0.021516695618629456, 0.033936675637960434, 0.03094293735921383, 0.03378918394446373, 
    0.044323261827230453, 0.034706637263298035, 0.03369896858930588, 0.03573163226246834, 0.02628341130912304, 
    0.014071882702410221, 0.0029828345868736506, 0.0015706412959843874, 0.0017107439925894141, 0.0014634154504165053, 
    0.004873405676335096, 0.002998138777911663, 0.0030240598134696484, 0.0009890805231407285, 0.0012279648799449205, 
    0.047324635088443756, 0.04472292214632034]

def init_model(args, lang_model, speaker_model, pose_dim, _device):
    # init model
    n_frames = args.n_poses
    generator = discriminator = audio_encoder = text_encoder = loss_fn = None
    if args.model == 'hierarchy':
        generator = Hierarchical_PoseGenerator(args,
                                  n_words=lang_model.n_words,
                                  word_embed_size=args.wordembed_dim,
                                  word_embeddings=lang_model.word_embedding_weights,
                                  z_obj=speaker_model,
                                  pose_dim=pose_dim)
        discriminator = Hierarchical_ConvDiscriminator(pose_dim)
        audio_encoder = Hierarchical_WavEncoder(args, z_obj=speaker_model, pose_level=6, nOut=32)
        text_encoder = TextEncoderTCN(args, lang_model.n_words, args.wordembed_dim, 
                pre_trained_embedding=lang_model.word_embedding_weights, dropout=args.dropout_prob)
    elif args.model == 'multimodal_context':
        generator = PoseGenerator(args,
                                  n_words=lang_model.n_words,
                                  word_embed_size=args.wordembed_dim,
                                  word_embeddings=lang_model.word_embedding_weights,
                                  z_obj=speaker_model,
                                  pose_dim=pose_dim)
        discriminator = ConvDiscriminator(pose_dim)
    elif args.model == 'joint_embedding':
        generator = EmbeddingNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                                 lang_model.word_embedding_weights, mode='random')
    elif args.model == 'gesture_autoencoder':
        generator = EmbeddingNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                                 lang_model.word_embedding_weights, mode='pose')
    elif args.model == 'seq2seq':
        generator = Seq2SeqNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                               lang_model.word_embedding_weights)
        loss_fn = torch.nn.L1Loss()
    elif args.model == 'speech2gesture':
        generator = speech2gesture.Generator(n_frames, pose_dim, args.n_pre_poses)
        discriminator = speech2gesture.Discriminator(pose_dim)
        loss_fn = torch.nn.L1Loss()

    return generator, discriminator, audio_encoder, text_encoder, loss_fn


def train_epochs(args, train_data_loader, test_data_loader, lang_model, pose_dim, speaker_model=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG'), AverageMeter('c_pos'), 
                   AverageMeter('c_neg'), AverageMeter('phy')]
    best_val_loss = (1e+10, 0)  # value, epoch

    tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_sample_result_epoch_interval = 10
    save_model_epoch_interval = 10

    # z type
    if args.z_type == 'speaker':
        pass
    elif args.z_type == 'random':
        speaker_model = 1
    else:
        speaker_model = None

    # init model
    g1, g2, g3, g4, g5, g6 = None, None, None, None, None, None
    generator, discriminator, audio_encoder, text_encoder, loss_fn = init_model(args, lang_model, speaker_model, pose_dim, device)
    if args.model == 'hierarchy':
        g1, _, _, _, _ = init_model(args, lang_model, speaker_model, 8 * 3, device)
        g2, _, _, _, _ = init_model(args, lang_model, speaker_model, 10 * 3, device)
        g3, _, _, _, _ = init_model(args, lang_model, speaker_model, 12 * 3, device)
        g4, _, _, _, _ = init_model(args, lang_model, speaker_model, 22 * 3, device)
        g5, _, _, _, _ = init_model(args, lang_model, speaker_model, 32 * 3, device)
        g6, _, _, _, _ = init_model(args, lang_model, speaker_model, 42 * 3, device)
        g1 = g1.to(device)
        g2 = g2.to(device)
        g3 = g3.to(device)
        g4 = g4.to(device)
        g5 = g5.to(device)
        g6 = g6.to(device)
        audio_encoder = audio_encoder.to(device)
        text_encoder = text_encoder.to(device)
    else:
        generator = generator.to(device)

    if discriminator is not None:
        discriminator = discriminator.to(device)

    # use multi GPUs
    if torch.cuda.device_count() > 1:
        if args.model == 'hierarchy':
            g1 = torch.nn.DataParallel(g1)
            g2 = torch.nn.DataParallel(g2)
            g3 = torch.nn.DataParallel(g3)
            g4 = torch.nn.DataParallel(g4)
            g5 = torch.nn.DataParallel(g5)
            g6 = torch.nn.DataParallel(g6)
            audio_encoder = torch.nn.DataParallel(audio_encoder)
            text_encoder = torch.nn.DataParallel(text_encoder)
        else:
            generator = torch.nn.DataParallel(generator)
        if discriminator is not None:
            discriminator = torch.nn.DataParallel(discriminator)

    # prepare an evaluator for FGD
    embed_space_evaluator = None
    if args.eval_net_path and len(args.eval_net_path) > 0:
        embed_space_evaluator = EmbeddingSpaceEvaluator(args, args.eval_net_path, lang_model, device)

    # define optimizers
    if args.model == 'hierarchy':
        # gen_optimizer = optim.Adam(list(g1.parameters()) + 
        #                             list(g2.parameters()) + 
        #                              list(g3.parameters()) + 
        #                               list(g4.parameters()) + 
        #                                list(g5.parameters()) + 
        #                                 list(g6.parameters())), lr=args.learning_rate, betas=(0.5, 0.999))
        gen_optimizer_1 = optim.Adam(g1.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        gen_optimizer_2 = optim.Adam(g2.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        gen_optimizer_3 = optim.Adam(g3.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        gen_optimizer_4 = optim.Adam(g4.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        gen_optimizer_5 = optim.Adam(g5.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        gen_optimizer_6 = optim.Adam(g6.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        audio_optimizer = optim.Adam(audio_encoder.parameters(),
                                         lr=args.learning_rate,
                                         betas=(0.5, 0.999))
        text_optimizer = optim.Adam(text_encoder.parameters(),
                                         lr=args.learning_rate,
                                         betas=(0.5, 0.999))
    else:
        gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    dis_optimizer = None
    if discriminator is not None:
        dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                         lr=args.learning_rate * args.discriminator_lr_weight,
                                         betas=(0.5, 0.999))

    # training
    global_iter = 0
    best_values = {}  # best values for all loss metrics
    for epoch in range(args.epochs):
        # evaluate the test set
        val_metrics = evaluate_testset(test_data_loader, generator, g1, g2, g3, g4, g5, g6, audio_encoder, loss_fn, embed_space_evaluator, args)

        # write to tensorboard and save best values
        for key in val_metrics.keys():
            tb_writer.add_scalar(key + '/validation', val_metrics[key], global_iter)
            if key != 'diversity':
                if key not in best_values.keys() or val_metrics[key] < best_values[key][0]:
                    best_values[key] = (val_metrics[key], epoch)
            else:
                if key not in best_values.keys() or val_metrics[key] > best_values[key][0]:
                    best_values[key] = (val_metrics[key], epoch) 

        # best?
        if 'frechet' in val_metrics.keys():
            val_loss = val_metrics['frechet']
        else:
            val_loss = val_metrics['loss']
        is_best = val_loss < best_val_loss[0]
        if is_best:
            logging.info('  *** BEST VALIDATION LOSS: {:.3f}'.format(val_loss))
            best_val_loss = (val_loss, epoch)
        else:
            logging.info('  best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        # save model
        if is_best or (epoch % save_model_epoch_interval == 0 and epoch > 0):
            dis_state_dict = None
            try:  # multi gpu
                if args.model == 'hierarchy':
                    gen_state_dict_1 = g1.module.state_dict()
                    gen_state_dict_2 = g2.module.state_dict()
                    gen_state_dict_3 = g3.module.state_dict()
                    gen_state_dict_4 = g4.module.state_dict()
                    gen_state_dict_5 = g5.module.state_dict()
                    gen_state_dict_6 = g6.module.state_dict()
                    audio_encoder_state_dict = audio_encoder.module.state_dict()
                    text_encoder_state_dict = text_encoder.module.state_dict()
                else:
                    gen_state_dict = generator.module.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.module.state_dict()
            except AttributeError:  # single gpu
                if args.model == 'hierarchy':
                    gen_state_dict_1 = g1.state_dict()
                    gen_state_dict_2 = g2.state_dict()
                    gen_state_dict_3 = g3.state_dict()
                    gen_state_dict_4 = g4.state_dict()
                    gen_state_dict_5 = g5.state_dict()
                    gen_state_dict_6 = g6.state_dict()
                    audio_encoder_state_dict = audio_encoder.state_dict()
                    text_encoder_state_dict = text_encoder.state_dict()
                else:
                    gen_state_dict = generator.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.state_dict()

            if is_best:
                save_name = '{}/{}_checkpoint_best.bin'.format(args.model_save_path, args.name)
            else:
                save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            if args.model == 'hierarchy':
                utils.train_utils.save_checkpoint({
                    'args': args, 'epoch': epoch, 'lang_model': lang_model, 'speaker_model': speaker_model,
                    'pose_dim': pose_dim, 'gen_dict_1': gen_state_dict_1, 'gen_dict_2': gen_state_dict_2, 'gen_dict_3': gen_state_dict_3,
                    'gen_dict_4': gen_state_dict_4, 'gen_dict_5': gen_state_dict_5, 'gen_dict_6': gen_state_dict_6,
                    'dis_dict': dis_state_dict, 'audio_dict': audio_encoder_state_dict, 'text_dict': text_encoder_state_dict
                }, save_name)
            else:
                utils.train_utils.save_checkpoint({
                    'args': args, 'epoch': epoch, 'lang_model': lang_model, 'speaker_model': speaker_model,
                    'pose_dim': pose_dim, 'gen_dict': gen_state_dict,
                    'dis_dict': dis_state_dict,
                }, save_name)

        # save sample results
        if args.save_result_video and epoch % save_sample_result_epoch_interval == 0:
            evaluate_sample_and_save_video(
                epoch, args.name, test_data_loader, generator,
                g1, g2, g3, g4, g5, g6, audio_encoder, 
                args=args, lang_model=lang_model)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            in_spec = (in_spec.float()).to(device)
            target_vec = target_vec.to(device)

            # speaker input
            vid_indices = []
            if speaker_model and isinstance(speaker_model, vocab.Vocab):
                vids = aux_info['vid']
                vid_indices = [speaker_model.word2index[vid] for vid in vids]
                vid_indices = torch.LongTensor(vid_indices).to(device)

            # train
            loss = []
            if args.model == 'hierarchy':
                loss = train_iter_hierarchy_expressive(args, epoch, in_text_padded, in_spec, target_vec, vid_indices,
                                      g1, g2, g3, g4, g5, g6, discriminator, audio_encoder, text_encoder,
                                      gen_optimizer_1, gen_optimizer_2, gen_optimizer_3, 
                                      gen_optimizer_4, gen_optimizer_5, gen_optimizer_6, dis_optimizer, 
                                      audio_optimizer, text_optimizer)
            elif args.model == 'multimodal_context':
                loss = train_iter_gan(args, epoch, in_text_padded, in_audio, target_vec, vid_indices,
                                      generator, discriminator,
                                      gen_optimizer, dis_optimizer)
            elif args.model == 'joint_embedding':
                loss = train_iter_embed(args, epoch, in_text_padded, in_audio, target_vec,
                                        generator, gen_optimizer, mode='random')
            elif args.model == 'gesture_autoencoder':
                loss = train_iter_embed(args, epoch, in_text_padded, in_audio, target_vec,
                                        generator, gen_optimizer)
            elif args.model == 'seq2seq':
                loss = train_iter_seq2seq(args, epoch, in_text, text_lengths, target_vec, generator, gen_optimizer)
            elif args.model == 'speech2gesture':
                loss = train_iter_speech2gesture(args, in_spec, target_vec, generator, discriminator,
                                                 gen_optimizer, dis_optimizer, loss_fn)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # write to tensorboard
            for key in loss.keys():
                tb_writer.add_scalar(key + '/train', loss[key], global_iter)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

    tb_writer.close()

    # print best losses
    logging.info('--------- best loss values ---------')
    for key in best_values.keys():
        logging.info('{}: {:.3f} at EPOCH {}'.format(key, best_values[key][0], best_values[key][1]))


def evaluate_testset(test_data_loader, generator, g1, g2, g3, g4, g5, g6, audio_encoder, loss_fn, embed_space_evaluator, args, sigma=0.1, thres=0.03):
    # to evaluation mode
    if args.model == 'hierarchy':
        g1.train(False)
        g2.train(False)
        g3.train(False)
        g4.train(False)
        g5.train(False)
        g6.train(False)
        audio_encoder.train(False)
    else:
        generator.train(False)

    if embed_space_evaluator:
        embed_space_evaluator.reset()
    losses = AverageMeter('loss')
    joint_mae = AverageMeter('mae_on_joint')
    accel = AverageMeter('accel')
    start = time.time()

    bc = AverageMeter('bc')
    beat_consistency_score = False

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            in_spec = (in_spec.float()).to(device)
            target = target_vec.to(device)

            # speaker input
            if args.model == 'hierarchy':
                try:
                    speaker_model = utils.train_utils.get_speaker_model(audio_encoder.feat_extractor)
                except AttributeError:
                    speaker_model = utils.train_utils.get_speaker_model(audio_encoder.module.feat_extractor)
            else:
                speaker_model = utils.train_utils.get_speaker_model(generator)
            if speaker_model:
                vid_indices = [random.choice(list(speaker_model.word2index.values())) for _ in range(batch_size)]
                vid_indices = torch.LongTensor(vid_indices).to(device)
            else:
                vid_indices = None

            pre_seq = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            pre_seq_partial = pre_seq[:, 0:args.n_pre_poses, :-1]

            if args.model == 'hierarchy':
                _, _, _, _, linear_blend_feat = audio_encoder(in_spec, vid_indices)

            target_1 = torch.cat((target[:, :, :3 * 3], target[:, :, -5 * 3:]), dim = 2)
            target_2 = torch.cat((target[:, :, :4 * 3], target[:, :, 20 * 3:21 * 3], target[:, :, -5 * 3:]), dim = 2)
            target_3 = torch.cat((target[:, :, :5 * 3], target[:, :, 20 * 3:22 * 3], target[:, :, -5 * 3:]), dim = 2)
            target_4 = torch.cat((target[:, :, :6 * 3], target[:, :, 8 * 3:9 * 3], target[:, :, 11 * 3:12 * 3], target[:, :, 14 * 3:15 * 3], target[:, :, 17 * 3:18 * 3], target[:, :, 20 * 3:23 * 3], target[:, :, 25 * 3:26 * 3], target[:, :, 28 * 3:29 * 3], target[:, :, 31 * 3:32 * 3], target[:, :, 34 * 3:35 * 3], target[:, :, -5 * 3:]), dim = 2)
            target_5 = torch.cat((target[:, :, :7 * 3], target[:, :, 8 * 3:10 * 3], target[:, :, 11 * 3:13 * 3], target[:, :, 14 * 3:16 * 3], target[:, :, 17 * 3:19 * 3], target[:, :, 20 * 3:24 * 3], target[:, :, 25 * 3:27 * 3], target[:, :, 28 * 3:30 * 3], target[:, :, 31 * 3:33 * 3], target[:, :, 34 * 3:36 * 3], target[:, :, -5 * 3:]), dim = 2)
            target_6 = target

            if args.model == 'joint_embedding':
                loss, out_dir_vec = eval_embed(in_text_padded, in_audio, pre_seq_partial,
                                               target, generator, mode='speech')
            elif args.model == 'gesture_autoencoder':
                loss, _ = eval_embed(in_text_padded, in_audio, pre_seq_partial, target, generator)
            elif args.model == 'seq2seq':
                out_dir_vec = generator(in_text, text_lengths, target, None)
                loss = loss_fn(out_dir_vec, target)
            elif args.model == 'speech2gesture':
                out_dir_vec = generator(in_spec, pre_seq_partial)
                loss = loss_fn(out_dir_vec, target)
            elif args.model == 'multimodal_context':
                out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid_indices)
                loss = F.l1_loss(out_dir_vec, target)
            elif args.model == 'hierarchy':
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

                loss = F.l1_loss(out_dir_vec, target_6)
            else:
                assert False

            losses.update(loss.item(), batch_size)

            if beat_consistency_score:
                beat_vec = out_dir_vec.cpu().numpy() + np.array(args.mean_dir_vec).squeeze()
                left_palm = torch.cross(beat_vec[:, :, 11 * 3 : 12 * 3], beat_vec[:, :, 17 * 3 : 18 * 3], dim = 2)
                right_palm = torch.cross(beat_vec[:, :, 28 * 3 : 29 * 3], beat_vec[:, :, 34 * 3 : 35 * 3], dim = 2)
                beat_vec = torch.cat((beat_vec, left_palm, right_palm), dim = 2)
                beat_vec = beat_vec.reshape(beat_vec.shape[0], beat_vec.shape[1], -1, 3)
                beat_vec = F.normalize(beat_vec, dim = -1)
                all_vec = beat_vec.reshape(beat_vec.shape[0] * beat_vec.shape[1], -1, 3)
                
                for idx, pair in enumerate(angle_pair):
                    vec1 = all_vec[:, pair[0]]
                    vec2 = all_vec[:, pair[1]]
                    inner_product = torch.einsum('ij,ij->i', [vec1, vec2])
                    inner_product = torch.clamp(inner_product, -1, 1, out=None)
                    angle = torch.acos(inner_product) / math.pi
                    angle_time = angle.reshape(batch_size, -1)
                    if idx == 0:
                        angle_diff = torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
                    else:
                        angle_diff += torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
                angle_diff = torch.cat((torch.zeros(batch_size, 1).to(device), angle_diff), dim = -1)
                
                for b in range(batch_size):
                    motion_beat_time = []
                    for t in range(2, 33):
                        if (angle_diff[b][t] < angle_diff[b][t - 1] and angle_diff[b][t] < angle_diff[b][t + 1]):
                            if (angle_diff[b][t - 1] - angle_diff[b][t] >= thres or angle_diff[b][t + 1] - angle_diff[b][t] >= thres):
                                motion_beat_time.append(float(t) / 15.0)
                    if (len(motion_beat_time) == 0):
                        continue
                    audio = in_audio[b].cpu().numpy()
                    audio_beat_time = librosa.onset.onset_detect(y=audio, sr=16000, units='time')
                    sum = 0
                    for audio in audio_beat_time:
                        sum += np.power(math.e, -np.min(np.power((audio - motion_beat_time), 2)) / (2 * sigma * sigma))
                    bc.update(sum / len(audio_beat_time), len(audio_beat_time))
            # print('evaluate bc: ', time.time() - test_start)
            if args.model != 'gesture_autoencoder':
                if embed_space_evaluator:
                    embed_space_evaluator.push_samples(in_text_padded, in_audio, out_dir_vec, target)

                # calculate MAE of joint coordinates
                out_dir_vec = out_dir_vec.cpu().numpy()
                out_dir_vec += np.array(args.mean_dir_vec).squeeze()
                out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)
                target_vec = target_vec.cpu().numpy()
                target_vec += np.array(args.mean_dir_vec).squeeze()
                target_poses = convert_dir_vec_to_pose(target_vec)

                if out_joint_poses.shape[1] == args.n_poses:
                    diff = out_joint_poses[:, args.n_pre_poses:] - target_poses[:, args.n_pre_poses:]
                else:
                    diff = out_joint_poses - target_poses[:, args.n_pre_poses:]
                mae_val = np.mean(np.absolute(diff))
                joint_mae.update(mae_val, batch_size)

                # accel
                target_acc = np.diff(target_poses, n=2, axis=1)
                out_acc = np.diff(out_joint_poses, n=2, axis=1)
                accel.update(np.mean(np.abs(target_acc - out_acc)), batch_size)

    # back to training mode
    if args.model == 'hierarchy':
        g1.train(True)
        g2.train(True)
        g3.train(True)
        g4.train(True)
        g5.train(True)
        g6.train(True)
        audio_encoder.train(True)
    else:
        generator.train(True)

    # print
    ret_dict = {'loss': losses.avg, 'joint_mae': joint_mae.avg}
    elapsed_time = time.time() - start
    if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
        frechet_dist, feat_dist = embed_space_evaluator.get_scores()
        diversity_score = embed_space_evaluator.get_diversity_scores()
        logging.info(
            '[VAL] loss: {:.3f}, joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, feat_D: {:.3f}, Diversity: {:.3f}, BC: {:.4f} / {:.1f}s'.format(
                losses.avg, joint_mae.avg, accel.avg, frechet_dist, feat_dist, diversity_score, bc.avg, elapsed_time))
        ret_dict['frechet'] = frechet_dist
        ret_dict['feat_dist'] = feat_dist
        ret_dict['diversity'] = diversity_score
        ret_dict['bc'] = bc.avg
    else:
        logging.info('[VAL] loss: {:.3f}, joint mae: {:.3f} / {:.1f}s'.format(
            losses.avg, joint_mae.avg, elapsed_time))

    return ret_dict


def evaluate_sample_and_save_video(epoch, prefix, test_data_loader, generator, 
                                   g1, g2, g3, g4, g5, g6, audio_encoder,
                                   args, lang_model,
                                   n_save=None, save_path=None):
    if args.model == 'hierarchy':
        g1.train(False)  # eval mode
        g2.train(False)  # eval mode
        g3.train(False)  # eval mode
        g4.train(False)  # eval mode
        g5.train(False)  # eval mode
        g6.train(False)  # eval mode
        audio_encoder.train(False)
    else:
        generator.train(False)  # eval mode
    start = time.time()
    if not n_save:
        n_save = 1 if epoch <= 0 else 5

    out_raw = []

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            if iter_idx >= n_save:  # save N samples
                break

            in_text, text_lengths, in_text_padded, _, target, in_audio, in_spec, aux_info = data

            # prepare
            select_index = 0
            if args.model == 'seq2seq':
                in_text = in_text[select_index, :].unsqueeze(0).to(device)
                text_lengths = text_lengths[select_index].unsqueeze(0).to(device)
            in_text_padded = in_text_padded[select_index, :].unsqueeze(0).to(device)
            in_audio = in_audio[select_index, :].unsqueeze(0).to(device)
            in_spec = in_spec[select_index, :, :].float().unsqueeze(0).to(device)
            target = target[select_index, :, :].unsqueeze(0).to(device)

            input_words = []
            for i in range(in_text_padded.shape[1]):
                word_idx = int(in_text_padded.data[select_index, i])
                if word_idx > 0:
                    input_words.append(lang_model.index2word[word_idx])
            sentence = ' '.join(input_words)

            # speaker input
            if torch.cuda.device_count() > 1:
                speaker_model = utils.train_utils.get_speaker_model(audio_encoder.module.feat_extractor)
            else:
                speaker_model = utils.train_utils.get_speaker_model(audio_encoder.feat_extractor)
            if speaker_model:
                vid = aux_info['vid'][select_index]
                # vid_indices = [speaker_model.word2index[vid]]
                vid_indices = [random.choice(list(speaker_model.word2index.values()))]
                vid_indices = torch.LongTensor(vid_indices).to(device)
            else:
                vid_indices = None

            # aux info
            aux_str = '({}, time: {}-{})'.format(
                aux_info['vid'][select_index],
                str(datetime.timedelta(seconds=aux_info['start_time'][select_index].item())),
                str(datetime.timedelta(seconds=aux_info['end_time'][select_index].item())))

            # synthesize
            pre_seq = target.new_zeros((target.shape[0], target.shape[1],
                                                target.shape[2] + 1))
            pre_seq[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            pre_seq_partial = pre_seq[:, 0:args.n_pre_poses, :-1]

            if args.model == 'hierarchy':
                _, _, _, _, linear_blend_feat = audio_encoder(in_spec, vid_indices)

            target_1 = torch.cat((target[:, :, :3 * 3], target[:, :, -5 * 3:]), dim = 2)
            target_2 = torch.cat((target[:, :, :4 * 3], target[:, :, 20 * 3:21 * 3], target[:, :, -5 * 3:]), dim = 2)
            target_3 = torch.cat((target[:, :, :5 * 3], target[:, :, 20 * 3:22 * 3], target[:, :, -5 * 3:]), dim = 2)
            target_4 = torch.cat((target[:, :, :6 * 3], target[:, :, 8 * 3:9 * 3], target[:, :, 11 * 3:12 * 3], target[:, :, 14 * 3:15 * 3], target[:, :, 17 * 3:18 * 3], target[:, :, 20 * 3:23 * 3], target[:, :, 25 * 3:26 * 3], target[:, :, 28 * 3:29 * 3], target[:, :, 31 * 3:32 * 3], target[:, :, 34 * 3:35 * 3], target[:, :, -5 * 3:]), dim = 2)
            target_5 = torch.cat((target[:, :, :7 * 3], target[:, :, 8 * 3:10 * 3], target[:, :, 11 * 3:13 * 3], target[:, :, 14 * 3:16 * 3], target[:, :, 17 * 3:19 * 3], target[:, :, 20 * 3:24 * 3], target[:, :, 25 * 3:27 * 3], target[:, :, 28 * 3:30 * 3], target[:, :, 31 * 3:33 * 3], target[:, :, 34 * 3:36 * 3], target[:, :, -5 * 3:]), dim = 2)
            target_6 = target

            if args.model == 'hierarchy':
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
            elif args.model == 'multimodal_context':
                out_dir_vec, *_ = generator(pre_seq, in_text_padded, in_audio, vid_indices)
            elif args.model == 'joint_embedding':
                _, _, _, _, _, _, out_dir_vec = generator(in_text_padded, in_audio, pre_seq_partial, None, 'speech')
            elif args.model == 'gesture_autoencoder':
                _, _, _, _, _, _, out_dir_vec = generator(in_text_padded, in_audio, pre_seq_partial, target,
                                                          variational_encoding=False)
            elif args.model == 'seq2seq':
                out_dir_vec = generator(in_text, text_lengths, target, None)
                # out_poses = torch.cat((pre_poses, out_poses), dim=1)
            elif args.model == 'speech2gesture':
                out_dir_vec = generator(in_spec, pre_seq_partial)

            # to video
            audio_npy = np.squeeze(in_audio.cpu().numpy())
            target = np.squeeze(target.cpu().numpy())
            out_dir_vec = np.squeeze(out_dir_vec.cpu().numpy())

            if save_path is None:
                save_path = args.model_save_path

            mean_data = np.array(args.mean_dir_vec).reshape(-1, 3)
            utils.train_utils.create_video_and_save(
                save_path, epoch, prefix, iter_idx,
                target, out_dir_vec, mean_data,
                sentence, audio=audio_npy, aux_str=aux_str)

            target = target.reshape((target.shape[0], args.pose_dim // 3, 3))
            out_dir_vec = out_dir_vec.reshape((out_dir_vec.shape[0], args.pose_dim // 3, 3))
            out_raw.append({
                'sentence': sentence,
                'audio': audio_npy,
                'human_dir_vec': target + mean_data,
                'out_dir_vec': out_dir_vec + mean_data,
                'aux_info': aux_str
            })

    if args.model == 'hierarchy':
        g1.train(True)
        g2.train(True)
        g3.train(True)
        g4.train(True)
        g5.train(True)
        g6.train(True)
        audio_encoder.train(True)
    else:
        generator.train(True)
    logging.info('saved sample videos, took {:.1f}s'.format(time.time() - start))

    return out_raw


def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

    # dataset config
    if args.model == 'seq2seq':
        collate_fn = word_seq_collate_fn
    else:
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
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
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

    # train
    # pose_dim = 126  # 42 x 3

    # Note that for TED Gesture    dataset, the pose dim is 27  (9 * 3)
    # While for the TED Expressive dataset, the pose dim is 126 (42 * 3)
    pose_dim = args.pose_dim
    train_epochs(args, train_loader, test_loader, lang_model,
                 pose_dim=pose_dim, speaker_model=train_dataset.speaker_model)


if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})