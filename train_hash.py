import argparse
import os
from datetime import datetime
import math

import fire
import numpy as np
from pprint import pprint
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils
from model import Model
from lars import LARS
import hash_utils
from hash import code_predict


def _momentum_update_key_encoder(model, model_k):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(model.parameters(), model_k.parameters()):
        param_k.data = param_k.data * args.m + param_q.data * (1. - args.m)

@torch.no_grad()
def _dequeue_and_enqueue(queue, queue_ptr, keys):
    batch_size = keys.shape[0]

    ptr = int(queue_ptr)
    assert args.K % batch_size == 0  # for simplicity

    # replace the keys at ptr (dequeue and enqueue)
    queue[:, ptr:ptr + batch_size] = keys.T
    ptr = (ptr + batch_size) % args.K  # move pointer

    queue_ptr[0] = ptr

# train for one epoch to learn unique features
def train(net, data_loader, optimizers, temperature, objective, p, schedulers):
    net.train()
    n_batch = len(data_loader)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for i, (pos_1, pos_2, target) in enumerate(train_bar):
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, middle_1, out_1, logit_1 = net(pos_1)
        if torch.cuda.device_count() > 1:
            # Shuffle examples
            idx_shuffle = torch.randperm(batch_size).cuda()
            pos_2 = pos_2[idx_shuffle]
            _, _, out_2, logit_2 = net(pos_2)
            idx_unshuffle = torch.argsort(idx_shuffle)
            out_2, logit_2 = out_2[idx_unshuffle], logit_2[idx_unshuffle]
        else:
            feature_2, middle_2, out_2, logit_2 = net(pos_2)
        if args.l2_warmup > 0:
            l2_weight = min(args.l2_weight, l2_weight + l2_weight / args.l2_warmup / n_batch)
        else:
            l2_weight = args.l2_weight

        if objective == 'nac':
            if args.nac_temperature or args.VI:
                temperature =  1 / (np.log((1 - p) / p) / 2)
            _out_1 = torch.tanh(out_1)
            _out_2 = torch.tanh(out_2)

            # Apply symmetric noise
            out = torch.cat([_out_1, _out_2], dim=0)
            m = (torch.rand_like(out) > p).to(torch.float32)
            m = 2 * (m - 0.5)
            out_drop = m * out

            # [2*B, 2*B]
            log_sim_matrix = torch.mm(out_drop, out.t().contiguous()) / temperature
            # mask = (torch.ones_like(log_sim_matrix) - torch.eye(2 * batch_size, device=log_sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            # log_sim_matrix = log_sim_matrix.masked_select(mask).view(2 * batch_size, -1)

            # compute loss
            if args.VI:
                # q_1, q_2: [B, D]
                q_1, q_2 = logit_1.sigmoid(), logit_2.sigmoid()
                log_pos_sim1 = (torch.sum(out_drop[:batch_size] * logit_2
                                        + torch.log(q_2 * (1 - q_2)) - np.log(1/4), dim=-1)) / 2
                log_pos_sim2 = (torch.sum(logit_1 * out_drop[batch_size:]
                                        + torch.log(q_1 * (1 - q_1)) - np.log(1/4), dim=-1)) / 2
            else:
                log_pos_sim1 = (torch.sum(out_drop[:batch_size] * _out_2, dim=-1)) / temperature
                log_pos_sim2 = (torch.sum(_out_1 * out_drop[batch_size:], dim=-1)) / temperature
            # [2*B]
            log_pos_sim = torch.cat([log_pos_sim1, log_pos_sim2], dim=0)
            loss = (- log_pos_sim
                    + torch.logsumexp(log_sim_matrix, dim=-1)).mean()
            loss += l2_weight * torch.sum(out ** 2, dim=-1).mean()
            if args.VI and args.ent_weight > 0:
                loss += args.ent_weight * (q_1 * q_1.log() + (1 - q_1) * (1 - q_1).log()
                                        + q_2 * q_2.log() + (1 - q_2) * (1 - q_2).log()
                                        + 2 * np.log(2)).sum(-1).mean()
        else:
            # SimCLR
            # [2*B, D]
            out_1 = F.normalize(out_1, dim=-1)
            out_2 = F.normalize(out_2, dim=-1)
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            log_sim_matrix = torch.mm(out, out.t().contiguous()) / temperature
            mask = (torch.ones_like(log_sim_matrix) - torch.eye(2 * batch_size, device=log_sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            log_sim_matrix = log_sim_matrix.masked_select(mask).view(2 * batch_size, -1)

            # compute loss
            log_pos_sim = torch.sum(out_1 * out_2, dim=-1) / temperature
            # [2*B]
            log_pos_sim = torch.cat([log_pos_sim, log_pos_sim], dim=0)
            # m = log_sim_matrix.max(dim=-1)[0]
            # loss = (- log_pos_sim
            #         + torch.log(torch.exp(log_sim_matrix - m.unsqueeze(1)).sum(dim=-1)) + m).mean()
            loss = (- log_pos_sim
                    + torch.logsumexp(log_sim_matrix, dim=-1)).mean()

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            if scheduler is not None:
                scheduler.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    # TODO: display images and labels
    with torch.no_grad():
        if scheduler is not None:
            writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('l2_weight', l2_weight, epoch)
        writer.add_scalar('out_norm', out_1.pow(2).sum(-1).mean().detach(), epoch)
        writer.add_histogram('out_norm_hist', out_1.pow(2).sum(-1).detach(), epoch)
        writer.add_scalar('feature_norm', feature_1.pow(2).sum(-1).mean().detach(), epoch)
        writer.add_histogram('feature_norm_hist', feature_1.pow(2).sum(-1).detach(), epoch)
        # Activation Codes
        writer.add_scalar('active_units', (out_1 > 0).to(torch.float32).mean().detach(), epoch)
        writer.add_histogram('active_units_hist', (out_1 > 0).to(torch.float32).mean(dim=-1).detach(), epoch)
        act_codes = torch.sign(out_1)
        hamming_dist = (act_codes.shape[-1]
                        - torch.mm(act_codes, act_codes.t().contiguous())) / 2
        mask = (torch.ones_like(hamming_dist) - torch.eye(batch_size, device=hamming_dist.device)).bool()
        hamming_dist = hamming_dist.masked_select(mask).view(batch_size, -1)
        writer.add_scalar('average_hamming_dist', hamming_dist.mean().detach(), epoch)
        writer.add_histogram('haming_dist_hist', hamming_dist.detach(), epoch)
        # Similarity
        out_1, out_2 = F.normalize(out_1, dim=-1), F.normalize(out_2, dim=-1)
        out = torch.cat([out_1, out_2], dim=0)
        out_sim = torch.mm(out, out.t().contiguous())
        writer.add_histogram('out_similarity', out_sim.detach(), epoch)
        feature = F.normalize(feature_1, dim=-1)
        feature_sim = torch.mm(feature, feature.t().contiguous())
        writer.add_histogram('feature_similarity', feature_sim.detach(), epoch)

        log_sim_matrix = out_sim
        mask = (torch.ones_like(log_sim_matrix) - torch.eye(2 * batch_size, device=log_sim_matrix.device)).bool()
        log_sim_matrix = log_sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        writer.add_scalar('average_similarity', log_sim_matrix.mean().detach(), epoch)
        log_pos_sim = torch.sum(out_1 * out_2, dim=-1)
        writer.add_scalar('positive_similarity', log_pos_sim.mean().detach(), epoch)
        writer.add_histogram('positive_similarity_hist', log_pos_sim.detach(), epoch)

        if args.VI:
            neg_entropy = (q_1 * q_1.log() + (1 - q_1) * (1 - q_1).log()
                           + np.log(2)).sum(-1).mean()
            writer.add_scalar('VI/neg_entropy', neg_entropy.detach(), epoch)
            logit_2 = F.normalize(logit_2, dim=-1)
            query_sim = torch.sum(out_1 * logit_2, dim=-1)
            writer.add_scalar('VI/query_similarity', query_sim.mean().detach(), epoch)
            writer.add_histogram('VI/query_similarity_hist', query_sim.detach(), epoch)
            writer.add_histogram('VI/q',
                                 q_1.detach(),
                                 epoch)

        writer.add_histogram('out', out_1.reshape(-1).detach(), epoch)

    return total_loss / total_num


def train_moco(net, net_k, queue, queue_ptr, data_loader, optimizers,
               temperature, objective, p, schedulers):
    net.train()
    net_k.train()
    n_batch = len(data_loader)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    l2_weight = args.l2_weight
    norm_out = None
    for i, (pos_1, pos_2, target) in enumerate(train_bar):
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                # Shuffle examples
                idx_shuffle = torch.randperm(batch_size).cuda()
                pos_2 = pos_2[idx_shuffle]
                feature_2, middle_2, out_2, logit_2 = net_k(pos_2)
                idx_unshuffle = torch.argsort(idx_shuffle)
                feature_2, middle_2, out_2, logit_2 = feature_2[idx_unshuffle], middle_2[idx_unshuffle], out_2[idx_unshuffle], logit_2[idx_unshuffle]
            else:
                feature_2, middle_2, out_2, logit_2 = net_k(pos_2)
        feature_1, middle_1, out_1, logit_1 = net(pos_1, feature_2)

        if objective == 'nac':
            if args.nac_temperature or args.VI:
                temperature =  1/ (np.log((1 - p) / p) / 2)
            if args.l2_normalize:
                _out_1 = F.normalize(out_1, dim=-1)
                _out_2 = F.normalize(out_2, dim=-1)
            else:
                _out_1 = torch.tanh(out_1)
                _out_2 = torch.tanh(out_2)
            # _out_1, _out_2 = F.normalize(_out_1, dim=-1), F.normalize(_out_2, dim=-1)
            # if not args.VI:
            #     _dequeue_and_enqueue(queue, queue_ptr, _out_2)
            if args.normalize_moco:
                _out_2 = _out_2 / out_2.norm(dim=1).mean()
            _dequeue_and_enqueue(queue, queue_ptr, _out_2)
            keys = queue.clone().detach()

            # Normalization
            if args.normalize_moco:
                if norm_out is None:
                    norm_out = _out_1.norm(dim=1).mean().detach()
                else:
                    norm_out = 0.9 * norm_out + 0.1 * _out_1.norm(dim=1).mean().detach()
                # norm_out = _out_1.norm(dim=1).mean().detach()
                norm_key = keys.norm(dim=0).mean()
                keys = keys * norm_out

            # if args.VI:
            #     # Include self in negatives
            #     keys = torch.cat([_out_1.t().contiguous(), keys], dim=1)

            # Apply symmetric noise
            m = (torch.rand_like(_out_1) > p).to(torch.float32)
            m = 2 * (m - 0.5)
            out_1_drop = m * _out_1

            # [B, K+1]
            log_sim_matrix = torch.mm(out_1_drop, keys) / temperature
            m = log_sim_matrix.max(dim=-1)[0]

            # compute loss
            if args.VI:
                # q_1: [B, D]
                q_1 = logit_1.sigmoid()
                log_pos_sim = (torch.sum(out_1_drop * logit_1
                                        + torch.log(q_1 * (1 - q_1)) - np.log(1/4), dim=-1)) / 2
                # _dequeue_and_enqueue(queue, queue_ptr, _out_2)
            else:
                # _out_2 = _out_2 * (norm_out / _out_2.norm(dim=1).mean())
                log_pos_sim = (torch.sum(out_1_drop * _out_2, dim=-1)) / temperature
            # [2*B]
            loss = (- log_pos_sim
                    + torch.log(torch.exp(log_sim_matrix - m.unsqueeze(1)).sum(dim=-1)) + m).mean()
            loss += l2_weight * torch.sum(_out_1 ** 2, dim=-1).mean()
            if args.VI and args.ent_weight > 0:
                loss += args.ent_weight * (q_1 * q_1.log() + (1 - q_1) * (1 - q_1).log()
                                           + np.log(2)).sum(-1).mean()
        else:
            # SimCLR
            # [2*B, D]
            out_1 = F.normalize(out_1, dim=-1)
            out_2 = F.normalize(out_2, dim=-1)
            _dequeue_and_enqueue(queue, queue_ptr, out_2)
            keys = queue.clone().detach()
            # [B, K]
            # keys = torch.cat([out_1.t().contiguous(), keys], dim=1)
            log_sim_matrix = torch.mm(out_1, keys) / temperature
            m = log_sim_matrix.max(dim=-1)[0]

            # compute loss
            log_pos_sim = torch.sum(out_1 * out_2, dim=-1) / temperature
            loss = (- log_pos_sim
                    + torch.log(torch.exp(log_sim_matrix - m.unsqueeze(1)).sum(dim=-1)) + m).mean()

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            if scheduler is not None:
                scheduler.step()
        _momentum_update_key_encoder(net, net_k)

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    with torch.no_grad():
        if args.normalize_moco:
            writer.add_scalar('norm_out', norm_out, epoch)
        if scheduler is not None:
            writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('l2_weight', l2_weight, epoch)
        writer.add_scalar('out_norm', out_1.pow(2).sum(-1).mean().detach(), epoch)
        writer.add_histogram('out_norm_hist', out_1.pow(2).sum(-1).detach(), epoch)
        writer.add_scalar('feature_norm', feature_1.pow(2).sum(-1).mean().detach(), epoch)
        writer.add_histogram('feature_norm_hist', feature_1.pow(2).sum(-1).detach(), epoch)
        # Activation Codes
        writer.add_scalar('active_units', (out_1 > 0).to(torch.float32).mean().detach(), epoch)
        writer.add_histogram('active_units_hist', (out_1 > 0).to(torch.float32).mean(dim=-1).detach(), epoch)
        act_codes = torch.sign(out_1)
        hamming_dist = (act_codes.shape[-1]
                        - torch.mm(act_codes, act_codes.t().contiguous())) / 2
        mask = (torch.ones_like(hamming_dist) - torch.eye(batch_size, device=hamming_dist.device)).bool()
        hamming_dist = hamming_dist.masked_select(mask).view(batch_size, -1)
        writer.add_scalar('average_hamming_dist', hamming_dist.mean().detach(), epoch)
        writer.add_histogram('haming_dist_hist', hamming_dist.detach(), epoch)
        # Similarity
        out_1, out_2 = F.normalize(out_1, dim=-1), F.normalize(out_2, dim=-1)
        keys = queue.clone().detach()
        writer.add_scalar('key_norm', keys.norm(dim=0).mean(), epoch)
        keys = F.normalize(keys, dim=-1)
        out_sim = torch.mm(out_1, keys)
        writer.add_histogram('out_similarity', out_sim.detach(), epoch)
        feature_sim = torch.mm(feature_1, feature_1.t().contiguous())
        writer.add_histogram('feature_similarity', feature_sim.detach(), epoch)
        log_sim_matrix = out_sim
        writer.add_scalar('average_similarity/out2key', log_sim_matrix.mean().detach(), epoch)
        out = torch.cat([out_1, out_2], dim=0)
        writer.add_scalar('average_similarity/out2out', torch.mm(out, out.t().contiguous()).mean().detach(), epoch)
        log_pos_sim = torch.sum(out_1 * out_2, dim=-1)
        writer.add_scalar('positive_similarity', log_pos_sim.mean().detach(), epoch)
        writer.add_histogram('positive_similarity_hist', log_pos_sim.detach(), epoch)

        if args.VI:
            neg_entropy = (q_1 * q_1.log() + (1 - q_1) * (1 - q_1).log()
                           + np.log(2)).sum(-1).mean()
            writer.add_scalar('VI/neg_entropy', neg_entropy.detach(), epoch)
            logit_1 = F.normalize(logit_1, dim=-1)
            query_sim = torch.sum(logit_1 * out_2, dim=-1)
            writer.add_scalar('VI/query_similarity', query_sim.mean().detach(), epoch)
            writer.add_histogram('VI/query_similarity_hist', query_sim.detach(), epoch)
            writer.add_histogram('VI/q',
                                 q_1.detach(),
                                 epoch)
    return total_loss / total_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--dataset', default='cifar', type=str, help='Datasets')
    parser.add_argument('--image_size', default=224, type=int, help='Image size')
    parser.add_argument('--optimizer', default='lars', type=str, help='Optimizer to use')
    parser.add_argument('--lr', default=3.0, type=float, help='Learning rate')
    parser.add_argument('--lr_warmup', default=10, type=int)
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=1000, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--architecture', type=str, default='resnet50')
    parser.add_argument('--objective', type=str, default='nac')
    parser.add_argument('--use_mrelu', action='store_true', default=False)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2_weight', default=1.0, type=float)
    parser.add_argument('--l2_warmup', default=0, type=int)
    parser.add_argument('--exclude_bias_decay', action='store_true', default=False)
    parser.add_argument('--exclude_bn_decay', action='store_true', default=False)
    parser.add_argument('--nac_temperature', action='store_true', default=False)
    parser.add_argument('--no_proj_bn', action='store_true', default=False)
    parser.add_argument('--VI', action='store_true', default=False)
    parser.add_argument('--ent_weight', default=0.0, type=float)
    parser.add_argument('--share_head', action='store_true', default=False)
    parser.add_argument('--use_last_output', action='store_true', default=False)
    parser.add_argument('--moco', action='store_true', default=False)
    parser.add_argument('--symmetric', action='store_true', default=False)
    parser.add_argument('--K', default=5000, type=int)
    parser.add_argument('--m', default=0.99, type=float)
    parser.add_argument('--normalize_moco', action='store_true', default=False)
    parser.add_argument('--l2_normalize', action='store_true', default=False)
    parser.add_argument('--l2_threshold', default=0.0, type=float)
    parser.add_argument('--note', type=str, default=None)
    parser.add_argument('--seed', default=0, type=int, help='seed for random variable')
    parser.add_argument('--n_query', default=10000, type=int, help='total number of queries to use')
    parser.add_argument('--finetune', default=None, type=float)

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    objective, dropout = args.objective, args.dropout
    pprint(vars(args))
    timestamp = datetime.now().strftime('%m-%d-%H:%M:%S')
    if args.objective == 'simclr':
        _args = [f'{key}={value}' for key, value in vars(args).items()
                        if key in ['dataset', 'objective', 'batch_size', 'lr', 'lr_warmup']]
    else:
        _args = [f'{key}={value}' for key, value in vars(args).items()
                 if key in ['feature_dim', 'architecture', 'image_size', 'objective', 'batch_size', 'lr', 'lr_warmup', 'l2_weight', 'dropout', 'optimizer', 'finetune']]

    _args.extend([f'{key}' for key, value in vars(args).items()
                  if key in ['exclude_bias_decay', 'exclude_bn_decay',
                             'VI', 'no_proj_bn', 'share_head', 'use_last_output',
                             'moco', 'symmetric', 'normalize_moco', 'l2_normalize'] and value])
    if args.moco:
        _args.append(f'K={args.K}.m={args.m}')
    if args.note is not None:
        _args.append(args.note)
    _args.insert(0, f'hash')

    exp_name = '.'.join((timestamp, *_args))
    save_dir = os.path.join(args.dataset, exp_name)

    # data prepare
    if args.dataset == 'cifar':
        train_data, query, gallery = hash_utils.get_cifar10pair_datasets('data', args.n_query, use_subset=True)
        print(len(train_data), len(query), len(gallery))
        n_gallery = len(gallery)
        query_loader = DataLoader(query, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        gallery_loader = DataLoader(gallery, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                  drop_last=True)
        R = n_gallery
        image_size = 32
        c = num_classes = 10
        multi_label = False
    elif args.dataset == 'flickr':
        train_data, query, gallery = hash_utils.get_flickr25kpair_datasets('/dataset_ssd_rubens/sangho.lee/mirflickr', '/dataset_ssd_rubens/sangho.lee/mirflickr25k_annotations_v080', args.image_size, 2000, 5000)
        n_gallery = len(gallery)
        query_loader = DataLoader(query, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        gallery_loader = DataLoader(gallery, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                  drop_last=True)
        R = n_gallery
        image_size = 224
        c = num_classes = 24
        multi_label = True
    elif args.dataset == 'nuswide':
        train_data, query, gallery = hash_utils.get_nuswidepair_datasets('/dataset_ssd_rubens/sangho.lee/nus_wide', '/dataset_ssd_rubens/sangho.lee/nus_wide_metas', args.image_size, 5000, 10500)
        n_gallery = len(gallery)
        query_loader = DataLoader(query, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        gallery_loader = DataLoader(gallery, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                  drop_last=True)
        R = n_gallery
        image_size = 224
        c = num_classes = 10
        multi_label = True

    # model setup and optimizer config
    model = Model(args.architecture, image_size, args.use_mrelu, feature_dim=feature_dim,
                  VI=args.VI, no_proj_bn=args.no_proj_bn, share_head=args.share_head,
                  use_last_output=args.use_last_output, pretrained=args.pretrained).cuda()
    if args.moco:
        model_k = Model(args.architecture, 32, args.use_mrelu, feature_dim=feature_dim,
                        VI=args.VI, no_proj_bn=args.no_proj_bn, share_head=args.share_head,
                        use_last_output=args.use_last_output).cuda()
        for param_q, param_k in zip(model.parameters(), model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        queue = 1e-3 * torch.randn(args.feature_dim, args.K).cuda()
        # queue = torch.zeros(args.feature_dim, args.K).cuda()
        if args.objective == 'simclr' or args.l2_normalize:
            queue = F.normalize(queue, dim=0)
        if args.normalize_moco:
            queue = queue / queue.norm(dim=0).mean()
        queue_ptr = torch.zeros(1, dtype=torch.long).cuda()


    # training loop
    best_acc = 0.0

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if args.moco:
            model_k = nn.DataParallel(model_k)

    def lr_schedule(step):
        num_samples = len(train_data) # assume cifar 10
        warmup_epochs = args.lr_warmup
        num_steps_per_epoch = num_samples // args.batch_size
        warmup_steps = num_steps_per_epoch * warmup_epochs
        total_steps = num_steps_per_epoch * args.epochs
        if step < warmup_steps:
            # Linear wamup for first n epochs
            factor = step / warmup_steps
        else:
            # Cosine learning rate schedule without restart
            factor = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        return factor


    def lr_schedule_no_warmup(step):
        num_samples = len(train_data) # assume cifar 10
        warmup_epochs = args.lr_warmup
        num_steps_per_epoch = num_samples // args.batch_size
        total_steps = num_steps_per_epoch * args.epochs
        # Cosine learning rate schedule without restart
        factor = 0.5 * (1 + math.cos(math.pi * (step) / (total_steps)))
        return factor

    if args.finetune is not None:
        module = model.module if torch.cuda.device_count() > 1 else model
        optimizer_base = LARS(module.encoder.parameters(), lr=args.finetune * args.lr,
                              weight_decay=args.weight_decay, weight_decay_params=None)
        if args.objective == 'nac':
            optimizer_head = LARS(list(module.head.parameters())
                                + list(module.linear.parameters())
                                + list(module.prediction.parameters()),
                                lr=args.lr,
                                weight_decay=args.weight_decay, weight_decay_params=None)
        else:
            optimizer_head = LARS(list(module.head.parameters())
                                + list(module.linear.parameters()),
                                lr=args.lr,
                                weight_decay=args.weight_decay, weight_decay_params=None)

        scheduler_base = optim.lr_scheduler.LambdaLR(optimizer_base, lr_schedule)
        scheduler_head = optim.lr_scheduler.LambdaLR(optimizer_head, lr_schedule_no_warmup)
        optimizers = [optimizer_base, optimizer_head]
        schedulers = [scheduler_base, scheduler_head]
    else:
        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = None
        else:
            if args.optimizer == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            elif args.optimizer == 'lars':
                if args.exclude_bias_decay and args.exclude_bn_decay:
                    weight_decay_params = [tensor for name, tensor in model.named_parameters()
                                            if 'bn' not in name and 'bias' not in name]
                elif args.exclude_bias_decay:
                    weight_decay_params = [tensor for name, tensor in model.named_parameters()
                                            if 'bias' not in name]
                elif args.exclude_bn_decay:
                    weight_decay_params = [tensor for name, tensor in model.named_parameters()
                                            if 'bn' not in name]
                else:
                    weight_decay_params = None
                optimizer = LARS(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, weight_decay_params=weight_decay_params)

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

        optimizers = [optimizer]
        schedulers = [scheduler]

    # For debugging
    step = np.array([0])
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir, flush_secs=10)

    query_hash, query_target = code_predict(model, query_loader, multi_label=multi_label)
    gallery_hash, gallery_target = code_predict(model, gallery_loader, multi_label=multi_label)
    code_and_label = {
        "gallery_hash": gallery_hash.numpy(),
        "gallery_target": gallery_target.numpy(),
        "query_hash": query_hash.numpy(),
        "query_target": query_target.numpy(),
    }
    mAP = hash_utils.mean_average_precision(code_and_label, R)
    print(f"mAP: {mAP:.3f}")
    writer.add_scalar('hashing/map', mAP, 0)

    for epoch in range(1, epochs + 1):
        if args.moco:
            train_loss = train_moco(model, model_k, queue, queue_ptr,
                                    train_loader, optimizers, temperature, objective, dropout, schedulers)
        else:
            train_loss = train(model, train_loader, optimizers, temperature, objective, dropout, schedulers)
        writer.add_scalar('pretraining/train_loss', train_loss, epoch)
        if epoch == epochs or epoch % 100 == 0:
            m = model.module if torch.cuda.device_count() > 1 else model
            torch.save(m.state_dict(), os.path.join(save_dir, f'last.pt'))

            query_hash, query_target = code_predict(model, query_loader, multi_label=multi_label)
            gallery_hash, gallery_target = code_predict(model, gallery_loader, multi_label=multi_label)
            code_and_label = {
                "gallery_hash": gallery_hash.numpy(),
                "gallery_target": gallery_target.numpy(),
                "query_hash": query_hash.numpy(),
                "query_target": query_target.numpy(),
            }
            mAP = hash_utils.mean_average_precision(code_and_label, R)
            print(f"mAP: {mAP:.3f}")
            writer.add_scalar('hashing/map', mAP, epoch)

    writer.flush()

# if __name__ == "__main__":
#     fire.Fire(run)
