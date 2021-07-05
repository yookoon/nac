from pyarrow import feather
import numpy as np
from sklearn import metrics
import os
import pickle
from pathlib import Path
from PIL import Image, ImageFilter
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from tqdm import tqdm
import utils


feat_cifar_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)


feat_mnist_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ]
)


standard_transform = transforms.Compose(
    [
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)


mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def mean_average_precision(params, R):
    query_hash = params['query_hash']
    gallery_hash = params['gallery_hash']
    query_target = params['query_target']
    gallery_target = params['gallery_target']
    query_num = query_hash.shape[0]

    sim = np.dot(gallery_hash, query_hash.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in tqdm(range(query_num), ncols=80):
        label = query_target[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(gallery_target[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R+1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)

    return np.mean(np.array(APx))


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        origin_index = self.indices[index]
        return self.dataset[origin_index]

    def __len__(self):
        return len(self.indices)


class HashDataset(torch.utils.data.Dataset):
    """Dataset for deep hashing
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        origin_index = self.indices[index]
        data, target = self.dataset[origin_index]
        return data, target

    def __len__(self):
        return len(self.indices)


class HashPairDataset(torch.utils.data.Dataset):
    """Dataset for deep hashing
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        origin_index = self.indices[index]
        data1, data2, target = self.dataset[origin_index]
        return data1, data2, target

    def __len__(self):
        return len(self.indices)


class FeatDataset(torch.utils.data.Dataset):
    """Feature Dataset
    """

    def __init__(self, feat_path):
        with open(feat_path, "rb") as f:
            data = pickle.load(f)
        self.feats = data['feat']
        self.labels = data['label']

    def __getitem__(self, index):
        return self.feats[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


@torch.no_grad()
def extract_cifar10_features(root, architecture):
    _architecture = architecture.split('_')[-1]
    feat_path = os.path.join(root, f"cifar10-{_architecture}.pkl")
    if not os.path.exists(feat_path):
        print("extract features...")
        model = getattr(models, f"{_architecture}")(pretrained=True)
        feat_extractor = nn.Sequential(*(list(model.children())[:-1]))

        trainset = CIFAR10(root=root, train=True, transform=feat_cifar_transform, download=True)
        testset = CIFAR10(root=root, train=False, transform=feat_cifar_transform, download=True)
        dataset = trainset + testset

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=100,
            shuffle=False,
            num_workers=16,
        )

        device = torch.cuda.current_device()
        feat_extractor = feat_extractor.cuda(device=device)
        feat_extractor.eval()

        all_feat = []
        all_label = []

        for images, labels in dataloader:
            images = images.to(device)
            feat = feat_extractor(images)
            feat = torch.flatten(feat, 1)
            all_feat.append(feat.detach().cpu())
            all_label.extend(labels.tolist())

        all_feat = torch.cat(all_feat, 0).numpy()
        data = {
            "feat": all_feat,
            "label": all_label,
        }
        with open(feat_path, "wb") as f:
            pickle.dump(data, f)


@torch.no_grad()
def extract_mnist_features(root, architecture):
    _architecture = architecture.split('_')[-1]
    feat_path = os.path.join(root, f"mnist-{_architecture}.pkl")
    if not os.path.exists(feat_path):
        print("extract features...")
        model = getattr(models, f"{_architecture}")(pretrained=True)
        feat_extractor = nn.Sequential(*(list(model.children())[:-1]))

        trainset = MNIST(root=root, train=True, transform=feat_mnist_transform, download=True)
        testset = MNIST(root=root, train=False, transform=feat_mnist_transform, download=True)
        dataset = trainset + testset

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=100,
            shuffle=False,
            num_workers=16,
        )

        device = torch.cuda.current_device()
        feat_extractor = feat_extractor.cuda(device=device)
        feat_extractor.eval()

        all_feat = []
        all_label = []

        for images, labels in dataloader:
            images = images.to(device)
            feat = feat_extractor(images)
            feat = torch.flatten(feat, 1)
            all_feat.append(feat.detach().cpu())
            all_label.extend(labels.tolist())

        all_feat = torch.cat(all_feat, 0).numpy()
        data = {
            "feat": all_feat,
            "label": all_label,
        }
        with open(feat_path, "wb") as f:
            pickle.dump(data, f)


@torch.no_grad()
def calculate_cifar10_distance(root, t, architecture, output_path):
    if 'pretrained' in architecture:
        _architecture = architecture.split('_')[-1]
        feat_path = os.path.join(root, f"cifar10-{_architecture}.pkl")
        dataset = FeatDataset(feat_path)
        dim = 1
    else:
        trainset = CIFAR10(root=root, train=True, transform=standard_transform, download=True)
        testset = CIFAR10(root=root, train=False, transform=standard_transform, download=True)
        dataset = trainset + testset
        dim = (1, 2, 3)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=16,
    )

    device = torch.cuda.current_device()

    # calculate the Euclidian distance to the t-th neighbor of each sample
    for j, data_t in enumerate(dataloader, 0):
        dist_list = []
        # get all inputs
        inputs_t, labels_t = data_t
        inputs_t, labels_t = inputs_t.to(device), labels_t.to(device)
        for i in range(len(inputs_t)):
            if i % 1000 == 0:
                print(i)
            aa = torch.mul(inputs_t - inputs_t[i], inputs_t - inputs_t[i])
            dist_m = torch.sqrt(torch.sum(aa, dim=dim))
            # dist_m[i] = 1000
            sorted_dist = np.sort(dist_m.detach().cpu().numpy())
            dist_list.append(sorted_dist[t])

    neighbor_dist = np.array(dist_list)
    np.savetxt(output_path, neighbor_dist)

    return neighbor_dist


@torch.no_grad()
def calculate_mnist_distance(root, t, architecture, output_path):
    if 'pretrained' in architecture:
        _architecture = architecture.split('_')[-1]
        feat_path = os.path.join(root, f"mnist-{_architecture}.pkl")
        dataset = FeatDataset(feat_path)
        dim = 1
    else:
        trainset = MNIST(root=root, train=True, transform=mnist_transform, download=True)
        testset = MNIST(root=root, train=False, transform=mnist_transform, download=True)
        dataset = trainset + testset
        dim = (1, 2, 3)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=16,
    )

    device = torch.cuda.current_device()

    # calculate the Euclidian distance to the t-th neighbor of each sample
    for j, data_t in enumerate(dataloader, 0):
        dist_list = []
        # get all inputs
        inputs_t, labels_t = data_t
        inputs_t, labels_t = inputs_t.to(device), labels_t.to(device)
        for i in range(len(inputs_t)):
            if i % 1000 == 0:
                print(i)
            aa = torch.mul(inputs_t - inputs_t[i], inputs_t - inputs_t[i])
            dist_m = torch.sqrt(torch.sum(aa, dim=dim))
            # dist_m[i] = 1000
            sorted_dist = np.sort(dist_m.detach().cpu().numpy())
            dist_list.append(sorted_dist[t])

    neighbor_dist = np.array(dist_list)
    np.savetxt(output_path, neighbor_dist)

    return neighbor_dist


def get_cifar10_datasets(root, architecture, n_query):
    if 'pretrained' in architecture:
        _architecture = architecture.split('_')[-1]
        feat_path = os.path.join(root, f"cifar10-{_architecture}.pkl")
        dataset = FeatDataset(feat_path)
    else:
        trainset = CIFAR10(root=root, train=True, transform=standard_transform, download=True)
        testset = CIFAR10(root=root, train=False, transform=standard_transform, download=True)
        dataset = trainset + testset

    perm = np.random.permutation(len(dataset))
    n_class = 10
    nsamples_per_class = n_query // n_class

    cnt_query = [0] * n_class
    ind_query = []
    ind_gallery = []

    for i in range(len(dataset)):
        _, label = dataset[perm[i]]
        if cnt_query[label] < nsamples_per_class:
            ind_query.append(perm[i])
            cnt_query[label] += 1
        else:
            ind_gallery.append(perm[i])

    data = HashDataset(dataset, list(range(len(dataset))))
    query = HashDataset(dataset, ind_query)
    gallery = HashDataset(dataset, ind_gallery)

    return data, query, gallery


def get_mnist_datasets(root, architecture, n_query):
    if 'pretrained' in architecture:
        _architecture = architecture.split('_')[-1]
        feat_path = os.path.join(root, f"mnist-{_architecture}.pkl")
        dataset = FeatDataset(feat_path)
    else:
        trainset = MNIST(root=root, train=True, transform=mnist_transform, download=True)
        testset = MNIST(root=root, train=False, transform=mnist_transform, download=True)
        dataset = trainset + testset

    perm = np.random.permutation(len(dataset))
    n_class = 10
    nsamples_per_class = n_query // n_class

    cnt_query = [0] * n_class
    ind_query = []
    ind_gallery = []

    for i in range(len(dataset)):
        _, label = dataset[perm[i]]
        if cnt_query[label] < nsamples_per_class:
            ind_query.append(perm[i])
            cnt_query[label] += 1
        else:
            ind_gallery.append(perm[i])

    data = HashDataset(dataset, list(range(len(dataset))))
    query = HashDataset(dataset, ind_query)
    gallery = HashDataset(dataset, ind_gallery)

    return data, query, gallery


def get_cifar10pair_datasets(root, n_query, seed=None, use_subset=False):
    if seed is not None:
        np.random.seed(seed)
    trainset = CIFAR10(root=root, train=True, transform=standard_transform, download=True)
    testset = CIFAR10(root=root, train=False, transform=standard_transform, download=True)
    dataset = trainset + testset

    perm = np.random.permutation(len(dataset))
    n_class = 10
    nsamples_per_class = n_query // n_class
    ntrain_per_class = 5000 // n_class if use_subset else 50000 // n_class

    cnt_query = [0] * n_class
    ind_query = []
    cnt_train = [0] * n_class
    ind_train = []
    ind_gallery = []

    for i in range(len(dataset)):
        _, label = dataset[perm[i]]
        if cnt_query[label] < nsamples_per_class:
            ind_query.append(perm[i])
            cnt_query[label] += 1
        else:
            ind_gallery.append(perm[i])
            if cnt_train[label] < ntrain_per_class:
                ind_train.append(perm[i])
                cnt_train[label] += 1


    pair_trainset = utils.CIFAR10Pair(root=root, train=True, transform=utils.train_transform, download=True)
    pair_testset = utils.CIFAR10Pair(root=root, train=False, transform=utils.train_transform, download=True)
    pair_dataset = pair_trainset + pair_testset

    train = HashPairDataset(pair_dataset, ind_train)
    query = HashDataset(dataset, ind_query)
    gallery = HashDataset(dataset, ind_gallery)

    return train, query, gallery


def to_cuda(array):
    device = torch.cuda.current_device()
    return torch.from_numpy(array).to(device)


def compute_metrics(params, N, R):
    query_hash = params['query_hash']
    gallery_hash = params['gallery_hash']

    query_target = params['query_target']
    gallery_target = params['gallery_target']

    withinN_precision_label = 0
    withinR_precision_label = 0

    mAP = 0

    n_query = len(query_target)

    for i in tqdm(range(n_query), ncols=80):
        # hamming_distance = (to_cuda(1 - query_hash[i]) == to_cuda(gallery_hash)).sum(axis=1)
        # hamming_distance = hamming_distance.detach().cpu().numpy()
        hamming_distance = np.sum((1 - query_hash[i]) == gallery_hash, axis=1)
        mAP += metrics.average_precision_score(
            gallery_target == query_target[i], 1. / (1 + hamming_distance)
        )

        nearestN_index = np.argsort(hamming_distance)[:N]
        withinN_precision_label += float(
            np.sum(gallery_target[nearestN_index] == query_target[i])
        ) / N

        withinR_label = gallery_target[hamming_distance < (R + 1)]
        num_withinR = len(withinR_label)
        if num_withinR > 0:
            withinR_precision_label += np.sum(
                withinR_label == query_target[i]
            ) / float(num_withinR)

    return mAP / n_query, withinN_precision_label / n_query, withinR_precision_label / n_query


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Flickr25K(torch.utils.data.Dataset):
    def __init__(self, data_dir, meta_dir, transform=None, target_transform=None):
        self.images = sorted(Path(data_dir).glob("*.jpg"))
        all_list = sorted(Path(meta_dir).glob("*.txt"))
        all_list = [path for path in all_list if path.name != "README.txt"]
        rel_list = sorted(Path(meta_dir).glob("*_r1.txt"))
        pot_list = [path for path in all_list if path not in rel_list]
        self.transform = transform
        self.target_transform = target_transform

        self._num_classes = len(pot_list)
        self._ind2label = defaultdict(list)
        for i, label_path in enumerate(pot_list):
            with open(label_path, "r") as f:
                ind_list = f.readlines()
            ind_list = [l.strip() for l in ind_list]
            for ind in ind_list:
                self._ind2label[int(ind)] += [i]

        self.class_names = [os.path.splitext(os.path.basename(label_path))[0]
                            for label_path in pot_list]

    def __getitem__(self, index):
        img_path = self.images[index]
        data = pil_loader(str(img_path))
        if self.transform is not None:
            data = self.transform(data)

        img_ind = int(img_path.stem[2:])
        label = self._ind2label[img_ind]
        target = [0] * self._num_classes
        for c_i in label:
            target[c_i] = 1

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.images)


class Flickr25KPair(Flickr25K):

    def __getitem__(self, index):
        img_path = self.images[index]
        data = pil_loader(str(img_path))
        if self.transform is not None:
            pos_1 = self.transform(data)
            pos_2 = self.transform(data)
        else:
            pos_1, pos_2 = data, data

        img_ind = int(img_path.stem[2:])
        label = self._ind2label[img_ind]
        target = [0] * self._num_classes
        for c_i in label:
            target[c_i] = 1

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class NusWide(torch.utils.data.Dataset):
    def __init__(self, data_dir, meta_dir, transform=None, target_transform=None):
        self._data_dir = data_dir
        with open(os.path.join(meta_dir, "images.txt"), "r") as f:
            image_list = f.readlines()
        self.images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = os.path.join(self._data_dir, self.images[index][0])
        data = pil_loader(str(img_path))
        if self.transform is not None:
            data = self.transform(data)

        target = self.images[index][1]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.images)


class NusWidePair(NusWide):

    def __getitem__(self, index):
        img_path = os.path.join(self._data_dir, self.images[index][0])
        data = pil_loader(str(img_path))
        if self.transform is not None:
            pos_1 = self.transform(data)
            pos_2 = self.transform(data)
        else:
            pos_1, pos_2 = data, data

        target = self.images[index][1]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def flickr_train_transform(image_size=224):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform


def flickr_test_transform(image_size=224):
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform


def get_flickr25kpair_datasets(data_dir, meta_dir, image_size, n_query, n_train, seed=None):
    if seed is not None:
        np.random.seed(seed)

    pair_dataset = Flickr25KPair(data_dir, meta_dir, flickr_train_transform(image_size))
    dataset = Flickr25K(data_dir, meta_dir, flickr_test_transform(image_size))

    perm = np.random.permutation(len(pair_dataset))
    ind_query = perm[:n_query]
    ind_gallary = perm[n_query:]
    ind_train = np.random.choice(ind_gallary, n_train, replace=False)

    data = IndexDataset(pair_dataset, ind_train)
    query = IndexDataset(dataset, ind_query)
    gallary = IndexDataset(dataset, ind_gallary)

    return data, query, gallary


def get_nuswidepair_datasets(data_dir, meta_dir, image_size, n_query, n_train, seed=None):
    if seed is not None:
        np.random.seed(seed)

    pair_dataset = NusWidePair(data_dir, meta_dir, flickr_train_transform(image_size))
    dataset = NusWide(data_dir, meta_dir, flickr_test_transform(image_size))

    perm = np.random.permutation(len(pair_dataset))
    ind_query = perm[:n_query]
    ind_gallary = perm[n_query:]
    ind_train = np.random.choice(ind_gallary, n_train, replace=False)

    data = IndexDataset(pair_dataset, ind_train)
    query = IndexDataset(dataset, ind_query)
    gallary = IndexDataset(dataset, ind_gallary)

    return data, query, gallary
