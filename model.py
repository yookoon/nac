import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torchvision.models import vgg16_bn

class Model(nn.Module):
    def __init__(self, feature_dim=128, VI=False, architecture='resnet50'):
        super(Model, self).__init__()

        # Modify the first conv layer according to SimCLR architecture
        if architecture == 'resnet50':
            f = []
            for name, module in resnet50().named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    f.append(module)
            # encoder
            self.encoder = nn.Sequential(*f)
            out_dim = 2048
        elif architecture == 'vgg16':
            self.encoder = vgg16_bn()
            out_dim = 512

        self.out_dim = out_dim

        self.head = nn.Sequential(nn.Linear(out_dim, out_dim, bias=True),
                                  nn.ReLU(inplace=True))
        self.linear = nn.Sequential(nn.Linear(out_dim, feature_dim, bias=True))

        self.VI = VI
        if VI:
            self.prediction = nn.Sequential(nn.Linear(out_dim, out_dim, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(out_dim, feature_dim, bias=True))

    def forward(self, x, feature_moco=None):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.head(feature)
        linear = self.linear(out)

        if self.VI:
            if feature_moco is not None:
                logit = self.prediction(feature_moco.detach())
            else:
                logit = self.prediction(feature)
        else:
            logit = torch.zeros_like(feature)

        return feature, linear / np.sqrt(linear.shape[-1]), logit / np.sqrt(logit.shape[-1])
