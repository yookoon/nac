from platform import architecture
import argparse
import os
from datetime import datetime
import math
from pathlib import Path

import numpy as np
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

import network
import utils
import hash_utils
from lars import LARS


def identity(x):
    return x


class Net(nn.Module):
    def __init__(self, architecture, image_size, hidden_list, n_bit):
        super(Net, self).__init__()

        if architecture in ['raw_cifar10']:
            self.encoder = identity
            encoder_dim = 3 * image_size * image_size
        elif architecture in ['raw_mnist']:
            self.encoder = identity
            encoder_dim = 1 * image_size * image_size
        else:
            if 'pretrained' in architecture:
                self.encoder = identity
            else:
                self.encoder = getattr(network, architecture)(image_size=image_size)

            if 'resnet50' in architecture:
                encoder_dim = 2048
            elif 'resnet18' in architecture or 'resnet34' in architecture:
                encoder_dim = 512
        self.encoder_dim = encoder_dim
        self.fc1 = nn.Linear(encoder_dim, hidden_list[0])
        self.fc2 = nn.Linear(hidden_list[0], hidden_list[1])
        self.fc3 = nn.Linear(hidden_list[1], n_bit)
        self.bn1 = nn.BatchNorm1d(hidden_list[0], eps=2e-5)
        self.bn1_F = nn.BatchNorm1d(hidden_list[0], eps=2e-5, affine=False)
        self.bn2 = nn.BatchNorm1d(hidden_list[1], eps=2e-5)
        self.bn2_F = nn.BatchNorm1d(hidden_list[1], eps=2e-5, affine=False)

    def forward(self, x, update_batch_stats=True):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        if update_batch_stats:
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x, inplace=True)
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x, inplace=True)
            x = self.fc3(x)
        else:
            x = self.fc1(x)
            x = self.bn1_F(x) * self.bn1.weight + self.bn1.bias
            x = F.relu(x, inplace=True)
            x = self.fc2(x)
            x = self.bn2_F(x) * self.bn2.weight + self.bn2.bias
            x = F.relu(x, inplace=True)
            x = self.fc3(x)
        return x


def binary_kl(p, q):
    # compute KL divergence between binary probablities p and q
    return torch.sum(
        p * torch.log((p + 1e-8) / (q + 1e-8))
        + (1 - p) * torch.log((1 - p + 1e-8) / (1 - q + 1e-8))
    ) / p.size(0)


def distance_sat(y0, y1):
    p0 = torch.sigmoid(y0)
    p1 = torch.sigmoid(y1)
    dist = binary_kl(p0, p1)
    return dist


def matrix_normalize(mat):
    sizes = mat.size()
    mat = mat.view(sizes[0], -1)
    mat = F.normalize(mat, p=2, dim=1)
    mat = mat.reshape(sizes)
    return mat


def vat(net, distance, x, prop_eps, eps_list, xi=10, Ip=1):
    # compute the regularized penality
    device = torch.cuda.current_device()

    with torch.no_grad():
        y = net(x, update_batch_stats=False)
    d = torch.randn_like(x)
    d = matrix_normalize(d)
    for ip in range(Ip):
        d = d.to(device)
        d.requires_grad_(True)
        y_p = net(x + xi * d, update_batch_stats=False)
        kl_loss = distance(y, y_p)
        kl_loss.backward(retain_graph=True)
        d = d.grad
        d = matrix_normalize(d)
    d = d.to(device)
    eps = prop_eps * eps_list
    sizes = [-1] + [1] * (x.dim() - 1)
    eps = eps.view(sizes)
    y_2 = net(x + eps * d, update_batch_stats=False)
    return distance(y, y_2)


def aug(net, distance, x1, x2):
    device = torch.cuda.current_device()
    y1 = net(x1)
    y2 = net(x2)
    return distance(y1, y2)


def loss_unlabeled(net, distance, x, prop_eps, eps_list):
    # compute the regularized penalty
    return vat(net, distance, x, prop_eps, eps_list)


def loss_information(net, x, n_bit):
    p_logit = net(x)
    p = torch.sigmoid(p_logit)
    p_ave = torch.sum(p, dim=0) / len(x)

    cond_ent = torch.sum(
        - p * torch.log(p + 1e-8) - (1 - p) * torch.log(1 - p + 1e-8)
    ) / p.size(0)
    marg_ent = torch.sum(
        - p_ave * torch.log(p_ave + 1e-8) - (1 - p_ave) * torch.log(1 - p_ave + 1e-8)
    )

    p_ave = p_ave.unsqueeze(0)

    p_ave_separated = torch.unbind(p_ave, dim=1)
    p_separated = torch.unbind(p.unsqueeze(2), dim=1)

    p_ave_list_i = []
    p_ave_list_j = []

    p_list_i = []
    p_list_j = []

    for i in range(n_bit - 1):
        p_ave_list_i.extend(list(p_ave_separated[i + 1:]))
        p_list_i.extend(list(p_separated[i + 1:]))

        p_ave_list_j.extend([p_ave_separated[i] for n in range(n_bit - i - 1)])
        p_list_j.extend([p_separated[i] for n in range(n_bit - i - 1)])

    p_ave_pair_i = torch.cat(p_ave_list_i, dim=0).unsqueeze(1)
    p_ave_pair_j = torch.cat(p_ave_list_j, dim=0).unsqueeze(1)

    p_pair_i = torch.cat(p_list_i, dim=1).unsqueeze(2)
    p_pair_j = torch.cat(p_list_j, dim=1).unsqueeze(2)

    p_pair_stacked_i = torch.cat(
        (p_pair_i, 1 - p_pair_i, p_pair_i, 1 - p_pair_i),
        dim=2,
    )
    p_pair_stacked_j = torch.cat(
        (p_pair_j, p_pair_j, 1 - p_pair_j, 1 - p_pair_j),
        dim=2,
    )

    p_ave_pair_stacked_i = torch.cat(
        (p_ave_pair_i, 1 - p_ave_pair_i, p_ave_pair_i, 1 - p_ave_pair_i),
        dim=1,
    )
    p_ave_pair_stacked_j = torch.cat(
        (p_ave_pair_j, p_ave_pair_j, 1 - p_ave_pair_j, 1 - p_ave_pair_j),
        dim=1,
    )

    p_product = torch.sum(p_pair_stacked_i * p_pair_stacked_j, dim=0) / len(p)
    p_ave_product = p_ave_pair_stacked_i * p_ave_pair_stacked_j
    pairwise_mi = 2 * torch.sum(p_product * torch.log((p_product + 1e-8) / (p_ave_product + 1e-8)))

    return cond_ent, marg_ent, pairwise_mi


@torch.no_grad()
def code_predict(net, loader):
    net.eval()
    device = torch.cuda.current_device()
    predict_bar = tqdm(loader, ncols=80)
    all_output = []
    all_label = []
    for images, labels, ind in predict_bar:
        images, labels = images.to(device), labels.to(device)
        output = torch.sigmoid(net(images)) > 0.5
        all_output.append(output.detach().cpu().float())
        all_label.append(labels.detach().cpu().float())
    all_output = torch.cat(all_output, 0)
    all_label = torch.cat(all_label, 0)

    return all_output, all_label


def train_vat(net, data_loader, train_optimizer, scheduler):
    net.train()
    n_batch = len(data_loader)
    total_loss, train_bar = 0.0, tqdm(data_loader, ncols=80)
    global_step = (epoch - 1) * n_batch
    for i, (inputs, targets, ind) in enumerate(train_bar):
        inputs = inputs.cuda(non_blocking=True)
        cond_ent, marg_ent, pairwise_mi = loss_information(net, inputs, n_bit)
        loss_info = cond_ent - marg_ent + pairwise_mi
        eps_list = neighbor_dist[ind]
        eps_list = eps_list.cuda(non_blocking=True)
        loss_ul = loss_unlabeled(net, distance_sat, inputs, args.prop_eps, eps_list)

        loss = loss_ul + args.lam * loss_info

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / (i + 1)))
        global_step += 1
        if global_step % 10 == 0:
            step_reg = loss_ul.item()
            step_mi = marg_ent.item() - cond_ent.item()
            step_pairwise_mi = pairwise_mi.item()
            step_loss = loss.item()
            writer.add_scalar('train/step_reg', step_reg, global_step)
            writer.add_scalar('train/step_mi', step_mi, global_step)
            writer.add_scalar('train/step_pairwise_mi', step_pairwise_mi, global_step)
            writer.add_scalar("train/step_loss", step_loss, global_step)

    return total_loss / n_batch


def train_aug(net, data_loader, train_optimizer, scheduler):
    net.train()
    n_batch = len(data_loader)
    total_loss, train_bar = 0.0, tqdm(data_loader, ncols=80)
    global_step = (epoch - 1) * n_batch
    for i, (input_1, input_2, target) in enumerate(train_bar):
        input_1, input_2 = input_1.cuda(non_blocking=True), input_2.cuda(non_blocking=True)

        cond_ent, marg_ent, pairwise_mi = loss_information(net, input_1, n_bit)
        loss_info = cond_ent - marg_ent + pairwise_mi
        loss_ul = aug(net, distance_sat, input_1, input_2)

        loss = loss_ul + args.lam * loss_info

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / (i + 1)))
        global_step += 1
        if global_step % 10 == 0:
            step_reg = loss_ul.item()
            step_mi = marg_ent.item() - cond_ent.item()
            step_pairwise_mi = pairwise_mi.item()
            step_loss = loss.item()
            writer.add_scalar('train/step_reg', step_reg, global_step)
            writer.add_scalar('train/step_mi', step_mi, global_step)
            writer.add_scalar('train/step_pairwise_mi', step_pairwise_mi, global_step)
            writer.add_scalar("train/step_loss", step_loss, global_step)

    return total_loss / n_batch


def test(net, query_loader, gallary_loader):
    query_hash, query_target = code_predict(net, query_loader)
    gallary_hash, gallary_target = code_predict(net, gallary_loader)
    params = {
        'query_hash': query_hash.numpy(),
        'query_target': query_target.numpy(),
        'gallary_hash': gallary_hash.numpy(),
        'gallary_target': gallary_target.numpy(),
    }

    return hash_utils.compute_metrics(params, args.N, args.R)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Hashing')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('--optimizer', default='lars', type=str, help='Optimizer to use')
    parser.add_argument('--lr', default=2.0, type=float, help='Learning rate')
    parser.add_argument('--lr_warmup', default=10, type=int)
    parser.add_argument('--lam', default=0.1, type=float,
                        help='trade-off parameter for mutual information and smooth regularization')
    parser.add_argument('--hidden_list', type=str, help='hidden size list', default='400-400')
    parser.add_argument('--n_bit', default=128, type=int, help='# of bits')
    parser.add_argument('--n_query', default=1000, type=int, help='# of query examples')
    parser.add_argument('--augmentation', default='vat', type=str, help='[vat, simclr]')
    parser.add_argument('--N', default=500, type=int)
    parser.add_argument('--R', default=2, type=int)
    parser.add_argument('--t', default=10, type=int, help='we use the distance to t-th neighbor for VAT')
    parser.add_argument('--prop_eps', default=0.25, type=float, help='epsilon for VAT')
    parser.add_argument('--batch_size', default=250, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--architecture', type=str, default='resnet50')
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--exclude_bias_decay', action='store_true', default=False)
    parser.add_argument('--exclude_bn_decay', action='store_true', default=False)
    parser.add_argument('--seed', default=0, type=int, help='seed for random variable')
    parser.add_argument('--note', type=str, default=None)

    # args parse
    args = parser.parse_args()
    batch_size, epochs = args.batch_size, args.epochs
    n_bit = args.n_bit
    t = args.t
    n_query = args.n_query
    hidden_list = list(map(int, args.hidden_list.split('-')))

    pprint(vars(args))
    timestamp = datetime.now().strftime('%m-%d-%H:%M:%S')
    _args = [
        f'{key}={value}' for key, value in vars(args).items()
        if key in ['seed', 'batch_size', 'lr', 'lr_warmup', 'augmentation']
    ]
    _args.extend(
        [
            f'{key}' for key, value in vars(args).items()
            if key in [
                'exclude_bias_decay',
                'exclude_bn_decay',
            ] and value
        ]
    )
    if args.note is not None:
        _args.append(args.note)

    exp_name = '.'.join((timestamp, *_args))
    save_dir = os.path.join(f'/home_greco/sangho.lee/{args.dataset}-imsat', exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # extract pretrained features
    if 'pretrained' in args.architecture:
        getattr(hash_utils, f"extract_{args.dataset}_features")('data', args.architecture)

    # data prepare
    print("Preparing data...")
    if args.augmentation == "vat":
        # import the range of local perturbation for VAT
        print("local perburbation for VAT...")
        suffix = 'raw' if 'pretrained' not in args.architecture else args.architecture.split('_')[-1]
        neighbor_path = os.path.join('data', f'{args.dataset}-{suffix}-{t}th_neighbor.txt')
        if os.path.exists(neighbor_path):
            neighbor_dist = torch.from_numpy(np.loadtxt(neighbor_path).astype(np.float32))
        else:
            neighbor_dist = torch.from_numpy(
                getattr(hash_utils, f'calculate_{args.dataset}_distance')('data', t, args.architecture, neighbor_path)
            )
        np.random.seed(args.seed)
        data, query, gallary = getattr(hash_utils, f'get_{args.dataset}_datasets')('data', args.architecture, n_query)
    elif args.augmentation == 'simclr':
        np.random.seed(args.seed)
        data, query, gallary = hash_utils.get_cifar10pair_datasets('data', n_query)
    n_gallary = len(gallary)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    query_loader = DataLoader(query, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    gallary_loader = DataLoader(gallary, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    image_size = 32 if args.dataset == 'cifar10' else 28

    # model setup and optimizer config
    model = Net(args.architecture, image_size, hidden_list, n_bit).cuda()

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

        def lr_schedule(step):
            num_samples = len(data) # assume cifar 10
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

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    writer = SummaryWriter(save_dir, flush_secs=10)
    best_map = 0.0

    for epoch in range(1, epochs + 1):
        if args.augmentation == 'vat':
            train_loss = train_vat(
                model, train_loader, optimizer, scheduler
            )
        elif args.augmentation == 'simclr':
            train_loss = train_aug(
                model, train_loader, optimizer, scheduler
            )
        writer.add_scalar('train/loss', train_loss, epoch)
        if epoch == epochs:
            torch.save(model.state_dict(), os.path.join(save_dir, f'last.pt'))
            mAP, withNpreclabel, withRpreclabel = test(
                model, query_loader, gallary_loader
            )
            print(f"FINAL EVAL | epoch: {epoch}, mAP: {mAP:.3f}, withNpreclabel: {withNpreclabel:.3f}, withRpreclabel: {withRpreclabel:.3f}")
            writer.add_scalar('eval/final_mAP', mAP, epoch)
            writer.add_scalar('eval/final_withNpreclabel', withNpreclabel, epoch)
            writer.add_scalar('eval/final_withRpreclabel', withRpreclabel, epoch)
        elif epoch % 1 == 0:
            mAP, withNpreclabel, withRpreclabel = test(
                model, query_loader, gallary_loader
            )
            print(f"EVAL | epoch: {epoch}, mAP: {mAP:.3f}, withNpreclabel: {withNpreclabel:.3f}, withRpreclabel: {withRpreclabel:.3f}")
            writer.add_scalar('eval/mAP', mAP, epoch)
            writer.add_scalar('eval/withNpreclabel', withNpreclabel, epoch)
            writer.add_scalar('eval/withRpreclabel', withRpreclabel, epoch)

    writer.flush()
