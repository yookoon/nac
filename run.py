import argparse
import os
from datetime import datetime
import math
from pprint import pprint
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import train_transform, linear_train_transform, test_transform, CIFAR10, CIFAR10Pair
from model import Model
from lars import LARS
from train import train, train_moco_symmetric

parser = argparse.ArgumentParser(description='Neural Activation Coding')
parser.add_argument('--objective', type=str, default='nac')
parser.add_argument('--optimizer', default='lars', type=str, help='Optimizer to use')
parser.add_argument('--lr', default=3.0, type=float, help='Learning rate')
parser.add_argument('--lr_warmup', default=10, type=int)
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument('--batch_size', default=1000, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--weight_decay', default=1e-6, type=float)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--exclude_bias_from_decay_params', action='store_true', default=False)
parser.add_argument('--exclude_bn_from_decay_params', action='store_true', default=False)
parser.add_argument('--moco', action='store_true', default=False)
parser.add_argument('--K', default=50000, type=int)
parser.add_argument('--m', default=0.99, type=float)


def linear_eval(network, feature_dim, num_classes, trainloader, testloader, use_sgd, lr, epochs, batch_size):
    linear = nn.Linear(feature_dim, num_classes)
    device = torch.cuda.current_device()
    network.eval()
    linear = linear.cuda(device=device)
    if use_sgd:
        optimizer = optim.SGD(linear.parameters(), lr=lr, momentum=0.9,
                              weight_decay=0.0, nesterov=True)
        num_steps_per_epoch = 50000 // batch_size
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
                feature, out, logit = network(images)
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
            feature, out, logit = network(images)
            outputs = linear(feature)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    test_accuracy = 100. * test_correct / test_total

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args()
    feature_dim, temperature = args.feature_dim, args.temperature
    batch_size, epochs = args.batch_size, args.epochs
    objective, dropout = args.objective, args.dropout
    optimizer, moco = args.optimizer, args.moco
    exclude_bias_from_decay_params = args.exclude_bias_from_decay_params
    exclude_bn_from_decay_params = args.exclude_bn_from_decay_params
    VI = (objective == 'nac')
    m, K = args.m, args.K
    pprint(vars(args))
    timestamp = datetime.now().strftime('%m-%d-%H:%M:%S')
    _args = [f'{key}={value}' for key, value in vars(args).items()
                    if key in ['objective', 'batch_size', 'lr', 'dropout']]
    _args.extend([f'{key}' for key, value in vars(args).items()
                  if key in ['exclude_bias_decay', 'exclude_bn_decay', 'moco'] and value])
    if moco:
        _args.append(f'K={K}.m={m}')

    exp_name = '.'.join((timestamp, *_args))
    save_dir = os.path.join('cifar', exp_name)

    # data prepare
    train_data = CIFAR10Pair(root='data', train=True,
                                   transform=train_transform,
                                   download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    memory_data = CIFAR10(root='data', train=True, transform=test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    train_data = CIFAR10(root='data', train=True, transform=linear_train_transform, download=True)
    linear_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_data = CIFAR10(root='data', train=False, transform=test_transform, download=True)
    linear_test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim=feature_dim, VI=VI).cuda()
    if moco:
        model_k = Model(feature_dim=feature_dim, VI=VI).cuda()
        for param_q, param_k in zip(model.parameters(), model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        queue = 1e-3 * torch.randn(feature_dim, K).cuda()
        if objective == 'simclr':
            queue = F.normalize(queue, dim=0)
        queue_ptr = torch.zeros(1, dtype=torch.long).cuda()

    num_classes = len(train_data.classes)
    linear_dim = model.out_dim
    model = nn.DataParallel(model)
    if moco:
        model_k = nn.DataParallel(model_k)

    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
    else:
        if optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
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
        if moco:
            train_loss = train_moco_symmetric(model, model_k, queue, queue_ptr,
                                              train_loader, optimizer, temperature,
                                              objective, dropout, scheduler, m, K, epoch, epochs)
        else:
            train_loss = train(model, train_loader, optimizer, temperature, objective, dropout, scheduler, epoch, epochs)
        writer.add_scalar('pretraining/train_loss', train_loss, epoch)
        if epoch == epochs:
            torch.save(model.module.state_dict(), os.path.join(save_dir, f'last.pt'))
            if moco:
                torch.save(model_k.module.state_dict(), os.path.join(save_dir, f'momentum.pt'))
            train_acc, test_acc = linear_eval(model,
                                              linear_dim,
                                              num_classes,
                                              linear_train_loader,
                                              linear_test_loader,
                                              True,
                                              1.0,
                                              100,
                                              batch_size)
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
                                              10,
                                              batch_size)
            print(f"LINEAR EVAL | epoch: {epoch}, train accuracy: {train_acc:.3f}, test accuracy: {test_acc:.3f}")
            writer.add_scalar('linear_eval/train_accuracy', train_acc, epoch)
            writer.add_scalar('linear_eval/test_accuracy', test_acc, epoch)

    writer.flush()