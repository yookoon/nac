#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from jedi.inference.gradual import conversion
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from datetime import datetime
from pprint import pprint
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import moco.loader
import moco.builder
from lars import LARS

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', default='nac_moco',
                    choices=['nac', 'nac_moco', 'moco'])
parser.add_argument('--optimizer', default='lars',
                    choices=['sgd', 'lars'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_warmup', default=0, type=int)
parser.add_argument('--schedule', default=None, nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# moco specific configs:
parser.add_argument('--dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--K', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

# configs for NAC
parser.add_argument('--p', default=0.1, type=float,
                    help='flip probability (default: 0.1)')
parser.add_argument('--l2_weight', default=0.1, type=float,
                    help='l2 regularization strength')
parser.add_argument('--VI', action='store_true',
                    help='Use variational inference')
parser.add_argument('--sync_batch_norm', action='store_true',
                    help='Use synchronized batch normalization')
parser.add_argument('--proj_bn', action='store_true',
                    help='Use projection head batch normalization')
parser.add_argument('--exclude_bn_bias', action='store_true',
                    help='Exclude batch norm and bias parameters from weight decay')

parser.add_argument('--debug', action='store_true',
                    help='Logging')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.cos = True
    args.multiprocessing_distributed = True
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.model == 'nac':
        model = moco.builder.NAC(
            models.__dict__[args.arch],
            args.dim, args.p, args.VI)
    elif args.model == 'nac_moco':
        model = moco.builder.NACMoCo(
            models.__dict__[args.arch],
            args.dim, args.K, args.m, args.p, args.VI, not args.sync_batch_norm,
            args.proj_bn)
    elif args.model == 'moco':
        model = moco.builder.MoCo(
            models.__dict__[args.arch],
            args.dim, args.K, args.m, args.t)
    # print(model)

    if args.sync_batch_norm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    if args.model == 'moco':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = None
    classification_criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    cudnn.benchmark = True

    # Data loading code
    traindir = 'train'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'lars':
        if args.exclude_bn_bias:
            weight_decay_params = [tensor for name, tensor in model.named_parameters()
                                    if 'bn' not in name and 'bias' not in name]
        else:
            weight_decay_params = None
        optimizer = LARS(model.parameters(), lr=args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay, weight_decay_params=weight_decay_params)

    def lr_schedule(step):
        warmup_epochs = args.lr_warmup
        num_steps_per_epoch = len(train_loader)
        warmup_steps = num_steps_per_epoch * warmup_epochs
        total_steps = num_steps_per_epoch * args.epochs
        step += num_steps_per_epoch * args.start_epoch
        if step < warmup_steps:
            # Linear wamup for first n epochs
            factor = step / warmup_steps
        else:
            # Cosine learning rate schedule without restart
            factor = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        return factor
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.gpu == 0:
        pprint(vars(args))
        timestamp = datetime.now().strftime('%m-%d-%H:%M:%S')
        exp_args = [f'{key}={value}' for key, value in vars(args).items()
                    if key in ['model', 'batch_size', 'world_size', 'lr',
                               'lr_warmup', 'l2_weight',
                               'K', 'm', 'weight_decay']]
        exp_args.extend([f'{key}' for key, value in vars(args).items()
                         if key in ['exclude_bn_bias',
                                    'no_proj_bn',
                                    'sync_batch_norm'] and value])
        exp_name = '.'.join((timestamp, *exp_args))
        save_dir = os.path.join('imagenet', exp_name)
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(save_dir, flush_secs=10)
    else:
        writer = None

    step = np.array([0])
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer,
              scheduler, step, classification_criterion)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(save_dir, '{:04d}.pth.tar'.format(epoch)))

    if writer is not None:
        writer.flush()


def train(train_loader, model, criterion, optimizer, epoch, args, writer,
          scheduler, step, classification_criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    norm = AverageMeter('Out Norm', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, norm],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    batch_size, p = args.batch_size, args.p

    end = time.time()
    for i, (images, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.model == 'moco':
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)
            out_norm = torch.tensor([1.0])
        elif args.model == 'nac':
            loss, out, output, target = model(im_1=images[0], im_2=images[1])
            loss = loss.mean()
            out_norm = torch.sum(out ** 2, dim=-1).mean()
            loss += args.l2_weight * out_norm
        elif args.model == 'nac_moco':
            loss, out, output, target, keys, logit_k, logit, l_pos, k, classifier_output = model(im_1=images[0], im_2=images[1])
            loss = loss.mean()
            out_norm = torch.sum(out ** 2, dim=-1).mean()
            loss += args.l2_weight * out_norm

            classification_loss = classification_criterion(classifier_output, label)
        else:
            raise NotImplementedError

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        losses.update(loss.item(), images[0].size(0))
        norm.update(out_norm.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        (loss + classification_loss).backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.model == 'nac_moco':
                k = F.normalize(k, dim=-1)
                keys = F.normalize(keys, dim=0)
                out = F.normalize(out, dim=-1)
                pos = torch.einsum('nc,nc->n', [out, k]).unsqueeze(-1)
                output = torch.mm(out, keys)
                output = torch.cat([pos, output], dim=1)

            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # top1.update(acc1[0], images[0].size(0))
            # top5.update(acc5[0], images[0].size(0))
            acc1, acc5 = accuracy(classifier_output, label, topk=(1, 5))
            top1.update(acc1[0], images[0].size(0))
            top5.update(acc5[0], images[0].size(0))

            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def kNN(net, k, temperature, memory_data_loader, test_data_loader, epoch, args, c=1000):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, _, out, _ = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, _, out, _ = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, args.epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    main()