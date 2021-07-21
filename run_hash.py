import argparse
import os
from datetime import datetime
import math
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Model
from lars import LARS
import hash_utils
from train import train, train_moco_symmetric

parser = argparse.ArgumentParser(description='Neural Activation Coding')
parser.add_argument('--objective', type=str, default='nac', choices=['nac', 'simclr'])
parser.add_argument('--optimizer', default='lars', type=str, help='Optimizer to use')
parser.add_argument('--lr', default=3.0, type=float, help='Learning rate')
parser.add_argument('--lr_warmup', default=10, type=int, help='Learning rate warmup epochs')
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature for SimCLR')
parser.add_argument('--batch_size', default=1000, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=2000, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--weight_decay', default=1e-6, type=float)
parser.add_argument('--flip', default=0.1, type=float, help='Flip probability in the noisy channel')
parser.add_argument('--exclude_bias_from_decay_params', action='store_true', default=False)
parser.add_argument('--exclude_bn_from_decay_params', action='store_true', default=False)
parser.add_argument('--moco', action='store_true', help='Whether to use momentum queue')
parser.add_argument('--K', default=5000, type=int, help='Size of momentum queue')
parser.add_argument('--m', default=0.99, type=float, help='Momentum queue decay')
parser.add_argument('--l2_weight', default=0.0, type=float, help='L2 regularization on the features')


if __name__ == "__main__":
    # args parse
    args = parser.parse_args()
    feature_dim, temperature = args.feature_dim, args.temperature
    batch_size, epochs = args.batch_size, args.epochs
    objective, flip = args.objective, args.flip
    optimizer, moco = args.optimizer, args.moco
    exclude_bias_from_decay_params = args.exclude_bias_from_decay_params
    exclude_bn_from_decay_params = args.exclude_bn_from_decay_params
    VI = (objective == 'nac')
    m, K = args.m, args.K
    weight_decay, l2_weight = args.weight_decay, args.l2_weight
    lr, lr_warmup = args.lr, args.lr_warmup
    pprint(vars(args))
    timestamp = datetime.now().strftime('%m-%d-%H:%M:%S')
    _args = [f'{key}={value}' for key, value in vars(args).items()
                    if key in ['objective', 'batch_size', 'lr', 'flip']]
    _args.extend([f'{key}' for key, value in vars(args).items()
                  if key in ['exclude_bias_from_decay_params', 'exclude_bn_from_decay_paramsecay', 'moco'] and value])
    if moco:
        _args.append(f'K={K}.m={m}')
    if moco:
        _args.append(f'K={K}.m={m}')
    save_dir = '.'.join(('cifar/hash', timestamp, *_args))

    # data prepare
    train_data, query, gallery = hash_utils.get_cifar10pair_datasets('data', 10000, use_subset=True)
    print(len(train_data), len(query), len(gallery))
    n_gallery = len(gallery)
    query_loader = DataLoader(query, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    gallery_loader = DataLoader(gallery, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    R = n_gallery
    num_classes = 10

    # model setup and optimizer config
    model = Model(feature_dim=feature_dim, VI=VI, architecture='vgg16').cuda()
    if moco:
        model_k = Model(feature_dim=feature_dim, VI=VI, architecture='vgg16').cuda()
        for param_q, param_k in zip(model.parameters(), model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        queue = 1e-3 * torch.randn(feature_dim, K).cuda()
        if objective == 'simclr':
            queue = F.normalize(queue, dim=0)
        queue_ptr = torch.zeros(1, dtype=torch.long).cuda()

    # training loop
    model = nn.DataParallel(model)
    if moco:
        model_k = nn.DataParallel(model_k)

    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = None
    else:
        if optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer == 'lars':
            if exclude_bias_from_decay_params and exclude_bn_from_decay_params:
                weight_decay_params = [tensor for name, tensor in model.named_parameters()
                                    if 'bn' not in name and 'bias' not in name]
            elif exclude_bias_from_decay_params:
                weight_decay_params = [tensor for name, tensor in model.named_parameters()
                                    if 'bias' not in name]
            elif exclude_bn_from_decay_params:
                weight_decay_params = [tensor for name, tensor in model.named_parameters()
                                    if 'bn' not in name]
            else:
                weight_decay_params = None
            optimizer = LARS(model.parameters(), lr=lr, weight_decay=weight_decay, weight_decay_params=weight_decay_params)

        def lr_schedule(step):
            num_samples = len(train_data)
            warmup_epochs = lr_warmup
            num_steps_per_epoch = num_samples // batch_size
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
        if moco:
            train_loss = train_moco_symmetric(model, model_k, queue, queue_ptr,
                                              train_loader, optimizer, temperature,
                                              objective, flip, scheduler, m, K,
                                              l2_weight, epoch, epochs)
        else:
            train_loss = train(model, train_loader, optimizer, temperature, objective, flip, scheduler, epoch, epochs)
        writer.add_scalar('pretraining/train_loss', train_loss, epoch)
        if epoch == epochs or epoch % 100 == 0:
            torch.save(model.module.state_dict(), os.path.join(save_dir, f'last.pt'))
            query_hash, query_target = hash_utils.code_predict(model, query_loader)
            gallery_hash, gallery_target = hash_utils.code_predict(model, gallery_loader)
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
