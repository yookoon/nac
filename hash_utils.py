import numpy as np
import random
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import utils


standard_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)


@torch.no_grad()
def code_predict(net, loader, num_classes=10):
    net.eval()
    device = torch.cuda.current_device()
    predict_bar = tqdm(loader, ncols=80)
    all_output = []
    all_label = []
    for images, labels in predict_bar:
        if type(labels) != torch.Tensor:
            labels = torch.stack(labels, dim=1)
        images, labels = images.to(device), labels.to(device)
        labels = F.one_hot(labels, num_classes)
        feature, middle, out, logit = net(images)
        all_output.append(out.detach().cpu().float())
        all_label.append(labels.detach().cpu().float())
    all_output = torch.cat(all_output, 0)
    all_label = torch.cat(all_label, 0)

    return (all_output > 0).float(), all_label


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
