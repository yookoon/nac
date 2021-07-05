import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import utils
from model import Model


class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.encoder = Model(args.feature_dim).encoder
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

        # missing_keys, unexpected_keys = self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

        d = torch.load(pretrained_path, map_location='cpu')
        s = {key[len('module.'):] if key.startswith('module.') else key: value for key, value in d.items()}
        missing_keys, unexpected_keys = self.load_state_dict(s, strict=False)
        assert missing_keys == ['fc.weight', 'fc.bias']

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

# train or test for one epoch
def train_val(net, data_loader, train_optimizer, scheduler):
    is_train = train_optimizer is not None
    # net.train() if is_train else net.eval()
    net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default=None,
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer to use')
    parser.add_argument('--lr', default=1.0, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')

    args = parser.parse_args()
    assert args.model_path is not None
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    train_data = CIFAR10(root='data', train=True, transform=utils.linear_train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_data = CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = Net(num_class=len(train_data.classes), pretrained_path=model_path).cuda()
    for param in model.encoder.parameters():
        param.requires_grad = False

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
        params = model.module.fc.parameters()
    else:
        params = model.fc.parameters()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay, nesterov=True)
        num_steps_per_epoch = 50000 // args.batch_size
        total_steps = num_steps_per_epoch * args.epochs
        def lr_schedule(step):
            # Cosine learning rate schedule without restart
            factor = 0.5 * (1 + math.cos(math.pi * step / total_steps))
            return factor
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer, scheduler)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, None)
        print(f'test_accuracy: {test_acc_1}')
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
