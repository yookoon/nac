import numpy as np
import random
import torch
import torch.nn.functional as F
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
def code_predict(net, loader, num_classes=10, multi_label=False):
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
        feature, out, logit = net(images)
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


def get_cifar10_datasets(root, n_query):
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
