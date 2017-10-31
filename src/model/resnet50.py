from collections import nametuple

import torch
from torchvision import models

class Resnet50(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet50, self).__init__()
        resnet_pretrained_features = models.resnet50(pretrained=True).features

        self.slice1 = torch.nn.Sequential()

        for x in range(138):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        out = h
        return out
