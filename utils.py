from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

linear_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

def _momentum_update_key_encoder(model, model_k, m):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(model.parameters(), model_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

@torch.no_grad()
def _dequeue_and_enqueue(queue, queue_ptr, keys, K):
    batch_size = keys.shape[0]

    ptr = int(queue_ptr)
    assert K % batch_size == 0  # for simplicity

    # replace the keys at ptr (dequeue and enqueue)
    queue[:, ptr:ptr + batch_size] = keys.T
    ptr = (ptr + batch_size) % K  # move pointer

    queue_ptr[0] = ptr

