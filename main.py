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

parser = argparse.ArgumentParser(description='Neural Activation Coding')
parser.add_argument('--optimizer', default='lars', type=str, help='Optimizer to use')
parser.add_argument('--lr', default=3.0, type=float, help='Learning rate')
parser.add_argument('--lr_warmup', default=10, type=int)
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument('--batch_size', default=1000, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--architecture', type=str, default='resnet50')
parser.add_argument('--objective', type=str, default='nac')
parser.add_argument('--weight_decay', default=1e-6, type=float)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--l2_warmup', default=0, type=int)
parser.add_argument('--exclude_from_decay_params', action='store_true', default=False)
parser.add_argument('--exclude_bn_decay_from_decay_params', action='store_true', default=False)
parser.add_argument('--moco', action='store_true', default=False)
parser.add_argument('--K', default=50000, type=int)
parser.add_argument('--m', default=0.99, type=float)

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, temperature, objective, p, scheduler):
    net.train()
    n_batch = len(data_loader)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for i, (pos_1, pos_2, target) in enumerate(train_bar):
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1, logit_1 = net(pos_1)
        if torch.cuda.device_count() > 1:
            # Shuffle examples
            idx_shuffle = torch.randperm(batch_size).cuda()
            pos_2 = pos_2[idx_shuffle]
            _, _, out_2, logit_2 = net(pos_2)
            idx_unshuffle = torch.argsort(idx_shuffle)
            out_2, logit_2 = out_2[idx_unshuffle], logit_2[idx_unshuffle]
        else:
            feature_2, out_2, logit_2 = net(pos_2)

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


def train_moco_symmetric(net, net_k, queue, queue_ptr, data_loader, train_optimizer,
                         temperature, objective, p, scheduler):
    net.train()
    net_k.train()
    n_batch = len(data_loader)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for i, (pos_1, pos_2, target) in enumerate(train_bar):
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        if torch.cuda.device_count() > 1:
            # Shuffle examples
            idx_shuffle = torch.randperm(batch_size).cuda()
            pos_2 = pos_2[idx_shuffle]
            feature_2, out_2, logit_2 = net(pos_2)
            idx_unshuffle = torch.argsort(idx_shuffle)
            feature_2, out_2, logit_2 = feature_2[idx_unshuffle], out_2[idx_unshuffle], logit_2[idx_unshuffle]
        else:
            feature_2, out_2, logit_2 = net(pos_2)
        feature_1, out_1, logit_1 = net(pos_1, feature_2)

        # Run momentum encoders
        with torch.no_grad():
            _, _, out_k_1, logit_k_1 = net_k(pos_1)
            _, _, out_k_2, logit_k_2 = net_k(pos_2)
        if torch.cuda.device_count() > 1:
            out_k_2, logit_k_2 = out_k_2[idx_unshuffle], logit_k_2[idx_unshuffle]

        # Apply symmetric loss
        if objective == 'nac':
            temperature =  1 / (np.log((1 - p) / p) / 2)
            _out_1, _out_2 = torch.tanh(out_1), torch.tanh(out_2)
            _out_k_1, _out_k_2 = torch.tanh(out_k_1), torch.tanh(out_k_2)
            out_k = torch.cat([_out_k_1, _out_k_2], dim=0)
            _dequeue_and_enqueue(queue, queue_ptr, out_k)
            keys = queue.clone().detach()

            # Apply symmetric noise
            out = torch.cat([_out_1, _out_2], dim=0)
            m = (torch.rand_like(out) > p).to(torch.float32)
            m = 2 * (m - 0.5)
            out_flip = m * out

            # [2B, K+2B]
            log_sim_matrix = torch.mm(out_flip, keys) / temperature

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
            _dequeue_and_enqueue(queue, queue_ptr, out_k_2)
            keys = queue.clone().detach()
            # [B, K]
            log_sim_matrix = torch.mm(out_1, keys) / temperature
            m = log_sim_matrix.max(dim=-1)[0]

            # compute loss
            log_pos_sim = torch.sum(out_1 * out_k_2, dim=-1) / temperature
            loss = (- log_pos_sim
                    + torch.log(torch.exp(log_sim_matrix - m.unsqueeze(1)).sum(dim=-1)) + m).mean()

            out_2 = F.normalize(out_2, dim=-1)
            out_k_1 = F.normalize(out_k_1, dim=-1)
            _dequeue_and_enqueue(queue, queue_ptr, out_k_1)
            keys = queue.clone().detach()
            # [B, K]
            log_sim_matrix = torch.mm(out_2, keys) / temperature
            m = log_sim_matrix.max(dim=-1)[0]

            # compute loss
            log_pos_sim = torch.sum(out_2 * out_k_1, dim=-1) / temperature
            loss += (- log_pos_sim
                     + torch.log(torch.exp(log_sim_matrix - m.unsqueeze(1)).sum(dim=-1)) + m).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        if scheduler is not None:
            scheduler.step()
        _momentum_update_key_encoder(net, net_k)

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

def linear_eval(network, feature_dim, num_classes, trainloader, testloader, use_sgd, lr, epochs):
    linear = nn.Linear(feature_dim, num_classes)
    device = torch.cuda.current_device()
    network.eval()
    linear = linear.cuda(device=device)

    if use_sgd:
        optimizer = optim.SGD(linear.parameters(), lr=lr, momentum=0.9,
                              weight_decay=0.0, nesterov=True)
        num_steps_per_epoch = 50000 // args.batch_size
        total_steps = num_steps_per_epoch * epochs
        def lr_schedule(step):
            # Cosine learning rate schedule without restart
            factor = 0.5 * (1 + math.cos(math.pi * step / total_steps))
            return factor
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    else:
        optimizer = optim.Adam(linear.parameters(), lr=lr, weight_decay=0.0)
        scheduler = None
    criterion = torch.nn.CrossEntropyLoss()
    epoch_bar = tqdm(range(1, epochs + 1))
    for epoch in epoch_bar:
        train_loss = 0
        train_correct = 0
        train_total = 0
        linear.train()
        for images, labels in trainloader:
            images, labels = images.cuda(device, non_blocking=True), labels.cuda(device, non_blocking=True)
            with torch.no_grad():
                feature, middle, out, r = network(images)
            outputs = linear(feature)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * train_correct / train_total

    test_loss = 0
    test_correct = 0
    test_total = 0
    linear.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            feature, middle, out, r = network(images)
            outputs = linear(feature)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    test_accuracy = 100. * test_correct / test_total

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    objective, dropout = args.objective, args.dropout
    pprint(vars(args))
    timestamp = datetime.now().strftime('%m-%d-%H:%M:%S')
    _args = [f'{key}={value}' for key, value in vars(args).items()
                    if key in ['objective', 'batch_size', 'lr', 'dropout']]
    _args.extend([f'{key}' for key, value in vars(args).items()
                  if key in ['exclude_bias_decay', 'exclude_bn_decay', 'moco'] and value])
    if args.moco:
        _args.append(f'K={args.K}.m={args.m}')

    exp_name = '.'.join((timestamp, *_args))
    save_dir = os.path.join('cifar', exp_name)

    # data prepare
    train_data = utils.CIFAR10Pair(root='data', train=True,
                                   transform=utils.train_transform,
                                   download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    memory_data = utils.CIFAR10(root='data', train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    train_data = utils.CIFAR10(root='data', train=True, transform=utils.linear_train_transform, download=True)
    linear_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_data = utils.CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    linear_test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    VI = args.objective == 'nac'
    model = Model(feature_dim=feature_dim, VI=VI).cuda()
    if args.moco:
        model_k = Model(feature_dim=feature_dim, VI=VI).cuda()
        for param_q, param_k in zip(model.parameters(), model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        queue = 1e-3 * torch.randn(args.feature_dim, args.K).cuda()
        if args.objective == 'simclr':
            queue = F.normalize(queue, dim=0)
        queue_ptr = torch.zeros(1, dtype=torch.long).cuda()

    num_classes = len(train_data.classes)
    linear_dim = model.out_dim
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if args.moco:
            model_k = nn.DataParallel(model_k)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
    else:
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optimizer == 'lars':
            if args.exclude_bias_from_decay_params and args.exclude_bn_from_decay_params:
                weight_decay_params = [tensor for name, tensor in model.named_parameters()
                                       if 'bn' not in name and 'bias' not in name]
            elif args.exclude_bias_from_decay_params:
                weight_decay_params = [tensor for name, tensor in model.named_parameters()
                                       if 'bias' not in name]
            elif args.exclude_bn_from_decay_params:
                weight_decay_params = [tensor for name, tensor in model.named_parameters()
                                       if 'bn' not in name]
            else:
                weight_decay_params = None
            optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, weight_decay_params=weight_decay_params)

        def lr_schedule(step):
            num_samples = 50000 # assume cifar 10
            warmup_epochs = args.lr_warmup
            num_steps_per_epoch = 50000 // batch_size
            warmup_steps = num_steps_per_epoch * warmup_epochs
            total_steps = num_steps_per_epoch * epochs
            if step < warmup_steps:
                # Linear wamup for first n epochs
                factor = step / warmup_steps
            else:
                # Cosine learning rate schedule without restart
                factor = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    step = np.array([0])
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir, flush_secs=10)
    for epoch in range(1, epochs + 1):
        if args.moco:
            train_loss = train_moco_symmetric(model, model_k, queue, queue_ptr,
                                                train_loader, optimizer, temperature, objective, dropout, scheduler)
        else:
            train_loss = train(model, train_loader, optimizer, temperature, objective, dropout, scheduler)
        writer.add_scalar('pretraining/train_loss', train_loss, epoch)
        if epoch == epochs:
            m = model.module if torch.cuda.device_count() > 1 else model
            torch.save(m.state_dict(), os.path.join(save_dir, f'last.pt'))
            if args.moco:
                m_k = model_k.module if torch.cuda.device_count() > 1 else model_k
                torch.save(m.state_dict(), os.path.join(save_dir, f'momentum.pt'))
            train_acc, test_acc = linear_eval(model,
                                              linear_dim,
                                              num_classes,
                                              linear_train_loader,
                                              linear_test_loader,
                                              True,
                                              1.0,
                                              100)
            print(f"FINAL LINEAR EVAL | epoch: {epoch}, train accuracy: {train_acc:.3f}, test accuracy: {test_acc:.3f}")
            writer.add_scalar('linear_eval/final_test_accuracy', test_acc, epoch)
        elif epoch % 10 == 0:
            train_acc, test_acc = linear_eval(model,
                                              linear_dim,
                                              num_classes,
                                              linear_train_loader,
                                              linear_test_loader,
                                              False,
                                              1e-1,
                                              10)
            print(f"LINEAR EVAL | epoch: {epoch}, train accuracy: {train_acc:.3f}, test accuracy: {test_acc:.3f}")
            writer.add_scalar('linear_eval/train_accuracy', train_acc, epoch)
            writer.add_scalar('linear_eval/test_accuracy', test_acc, epoch)

    writer.flush()
