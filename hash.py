import argparse
import os
from datetime import datetime
import math

import numpy as np
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model
import hash_utils


@torch.no_grad()
def code_predict(net, loader, num_classes=10, use_default_train=False,
                 multi_label=False):
    net.eval()
    device = torch.cuda.current_device()
    predict_bar = tqdm(loader, ncols=80)
    all_output = []
    all_label = []
    for images, labels in predict_bar:
        if type(labels) != torch.Tensor:
            labels = torch.stack(labels, dim=1)
        images, labels = images.to(device), labels.to(device)
        if multi_label:
            pass
        else:
            labels = F.one_hot(labels, num_classes)
        feature, middle, out, logit = net(images)
        all_output.append(out.detach().cpu().float())
        all_label.append(labels.detach().cpu().float())
    all_output = torch.cat(all_output, 0)
    all_label = torch.cat(all_label, 0)

    return (all_output > 0).float(), all_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Hashing')
    parser.add_argument("--model_path", type=str, default=None, help="The pretrained model path")
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--batch_size', default=1000, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--architecture', type=str, default='resnet50')
    parser.add_argument('--use_mrelu', action='store_true', default=False)
    parser.add_argument('--no_proj_bn', action='store_true', default=False)
    parser.add_argument('--VI', action='store_true', default=False)
    parser.add_argument('--share_head', action='store_true', default=False)
    parser.add_argument('--seed', default=0, type=int, help='seed for random variable')
    parser.add_argument('--n_query', default=1000, type=int, help='# of query examples')
    parser.add_argument('--N', default=500, type=int)
    parser.add_argument('--R', default=50000, type=int)
    parser.add_argument('--use_default_train', action='store_true', default=False)
    parser.add_argument('--note', type=str, default=None)

    # args parse
    args = parser.parse_args()
    feature_dim, R = args.feature_dim, args.R
    model_path, batch_size = args.model_path, args.batch_size
    n_query = args.n_query
    pprint(vars(args))

    # data prepare
    print("Preparing data...")
    if args.use_default_train:
        gallery_data = CIFAR10(root='data', train=True, transform=utils.test_transform, download=True)
        gallery_loader = DataLoader(gallery_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        query_data = CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
        query_loader = DataLoader(query_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    else:
        np.random.seed(args.seed)
        data, query, gallery = hash_utils.get_cifar10_datasets('data', '', n_query)
        n_gallery = len(gallery)
        query_loader = DataLoader(query, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        gallery_loader = DataLoader(gallery, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    print("*"*50)
    print(f"Deep hashing with {feature_dim} bits, Retrieval (R): {R}")

    # model setup and optimizer config
    model = Model(args.architecture, 32, args.use_mrelu, feature_dim=feature_dim,
                  VI=args.VI, no_proj_bn=args.no_proj_bn, share_head=args.share_head).cuda()

    if model_path is not None:
        d = torch.load(model_path, map_location='cpu')
        model.load_state_dict(d)
        print(f"checkpoint: {model_path}")
    else:
        print(f"Random initialization")

    print(f"gallery hash coding...")
    gallery_hash, gallery_target = code_predict(model, gallery_loader, use_default_train=args.use_default_train)
    print("Query hash coding...")
    query_hash, query_target = code_predict(model, query_loader, use_default_train=args.use_default_train)
    code_and_label = {
        "gallery_hash": gallery_hash.numpy(),
        "gallery_target": gallery_target.numpy(),
        "query_hash": query_hash.numpy(),
        "query_target": query_target.numpy(),
    }
    print("Computing metrics...")
    # mAP, withNpreclabel, withRpreclabel = hash_utils.compute_metrics(code_and_label, args.N, args.R)
    # print(f"mAP: {mAP:.3f}, withNpreclabel: {withNpreclabel:.3f}, withRpreclabel: {withRpreclabel:.3f}")
    mAP = hash_utils.mean_average_precision(code_and_label, args.R)
    print(f"mAP: {mAP:.3f}")
