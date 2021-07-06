import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import _momentum_update_key_encoder, _dequeue_and_enqueue

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, temperature, objective, p, scheduler, epoch, epochs):
    net.train()
    n_batch = len(data_loader)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for i, (pos_1, pos_2, target) in enumerate(train_bar):
        batch_size = len(pos_1)
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1, logit_1 = net(pos_1)
        # Batch shuffling
        idx_shuffle = torch.randperm(batch_size).cuda()
        pos_2 = pos_2[idx_shuffle]
        _, out_2, logit_2 = net(pos_2)
        idx_unshuffle = torch.argsort(idx_shuffle)
        out_2, logit_2 = out_2[idx_unshuffle], logit_2[idx_unshuffle]

        if objective == 'nac':
            _out_1 = torch.tanh(out_1)
            _out_2 = torch.tanh(out_2)

            # Apply symmetric noise
            out = torch.cat([_out_1, _out_2], dim=0)
            m = (torch.rand_like(out) > p).to(torch.float32)
            m = 2 * (m - 0.5)
            out_flip = m * out

            # [2*B, 2*B]
            log_sim_matrix = torch.mm(out_flip, out.t().contiguous()) / (np.log((1 - p) / p) / 2)

            # q_1, q_2: [B, D]
            q_1, q_2 = logit_1.sigmoid(), logit_2.sigmoid()
            log_pos_sim1 = (torch.sum(out_flip[:batch_size] * logit_2
                                    + torch.log(q_2 * (1 - q_2)) - np.log(1/4), dim=-1)) / 2
            log_pos_sim2 = (torch.sum(logit_1 * out_flip[batch_size:]
                                    + torch.log(q_1 * (1 - q_1)) - np.log(1/4), dim=-1)) / 2
            # [2*B]
            log_pos_sim = torch.cat([log_pos_sim1, log_pos_sim2], dim=0)
            loss = (- log_pos_sim + torch.logsumexp(log_sim_matrix, dim=-1)).mean()
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
            loss = (- log_pos_sim
                    + torch.logsumexp(log_sim_matrix, dim=1)).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    return total_loss / total_num

def train_moco_symmetric(net, net_k, queue, queue_ptr, data_loader, train_optimizer,
                         temperature, objective, p, scheduler, moco_m, moco_K, l2_weight,
                         epoch, epochs):
    net.train()
    net_k.train()
    n_batch = len(data_loader)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for i, (pos_1, pos_2, target) in enumerate(train_bar):
        batch_size = len(pos_1)
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        # Batch shuffling
        idx_shuffle = torch.randperm(batch_size).cuda()
        pos_2 = pos_2[idx_shuffle]
        feature_2, out_2, logit_2 = net(pos_2)
        idx_unshuffle = torch.argsort(idx_shuffle)
        feature_2, out_2, logit_2 = feature_2[idx_unshuffle], out_2[idx_unshuffle], logit_2[idx_unshuffle]
        feature_1, out_1, logit_1 = net(pos_1)

        # Run momentum encoders
        with torch.no_grad():
            _, out_k_1, logit_k_1 = net_k(pos_1)
            _, out_k_2, logit_k_2 = net_k(pos_2)
            out_k_2, logit_k_2 = out_k_2[idx_unshuffle], logit_k_2[idx_unshuffle]

        # Apply symmetric loss
        if objective == 'nac':
            _out_1, _out_2 = torch.tanh(out_1), torch.tanh(out_2)
            _out_k_1, _out_k_2 = torch.tanh(out_k_1), torch.tanh(out_k_2)
            out_k = torch.cat([_out_k_1, _out_k_2], dim=0)
            _dequeue_and_enqueue(queue, queue_ptr, out_k, moco_K)
            keys = queue.clone().detach()

            # Apply symmetric noise
            out = torch.cat([_out_1, _out_2], dim=0)
            m = (torch.rand_like(out) > p).to(torch.float32)
            m = 2 * (m - 0.5)
            out_flip = m * out

            # [2B, K+2B]
            log_sim_matrix = torch.mm(out_flip, keys) / (np.log((1 - p) / p) / 2)

            # q_1, q_2: [B, D]
            q_k_1, q_k_2 = logit_k_1.sigmoid(), logit_k_2.sigmoid()
            log_pos_sim1 = (torch.sum(out_flip[:batch_size] * logit_k_2
                                    + torch.log(q_k_2 * (1 - q_k_2)) - np.log(1/4), dim=-1)) / 2
            log_pos_sim2 = (torch.sum(logit_k_1 * out_flip[batch_size:]
                                    + torch.log(q_k_1 * (1 - q_k_1)) - np.log(1/4), dim=-1)) / 2
            # [2*B]
            log_pos_sim = torch.cat([log_pos_sim1, log_pos_sim2], dim=0)
            loss = (- log_pos_sim
                    + torch.logsumexp(log_sim_matrix, dim=-1)).mean()
            # L2 regularization on features
            loss += l2_weight * torch.sum(out ** 2, dim=-1).mean()
            q_1, q_2 = logit_1.sigmoid(), logit_2.sigmoid()
            log_pos_sim1_VI = (torch.sum(out_flip[:batch_size].detach() * logit_2
                                    + torch.log(q_2 * (1 - q_2)) - np.log(1/4), dim=-1)) / 2
            log_pos_sim2_VI = (torch.sum(logit_1 * out_flip[batch_size:].detach()
                                    + torch.log(q_1 * (1 - q_1)) - np.log(1/4), dim=-1)) / 2
            loss = loss - log_pos_sim1_VI.mean() - log_pos_sim2_VI.mean()
        else:
            # SimCLR
            out_1 = F.normalize(out_1, dim=-1)
            out_k_2 = F.normalize(out_k_2, dim=-1)
            _dequeue_and_enqueue(queue, queue_ptr, out_k_2, moco_K)
            keys = queue.clone().detach()
            # [B, K]
            log_sim_matrix = torch.mm(out_1, keys) / temperature

            # compute loss
            log_pos_sim = torch.sum(out_1 * out_k_2, dim=-1) / temperature
            loss = (- log_pos_sim
                    + torch.logsumexp(log_sim_matrix, dim=1)).mean()

            out_2 = F.normalize(out_2, dim=-1)
            out_k_1 = F.normalize(out_k_1, dim=-1)
            _dequeue_and_enqueue(queue, queue_ptr, out_k_1, moco_K)
            keys = queue.clone().detach()
            # [B, K]
            log_sim_matrix = torch.mm(out_2, keys) / temperature

            # compute loss
            log_pos_sim = torch.sum(out_2 * out_k_1, dim=-1) / temperature
            loss += (- log_pos_sim
                     + torch.logsumexp(log_sim_matrix, dim=1)).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        if scheduler is not None:
            scheduler.step()
        _momentum_update_key_encoder(net, net_k, moco_m)

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num
