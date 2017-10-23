from collections import nametuple

import torch
from torchvision import models

class Resnet50(torhc.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet50, self).__init__()
        resnet_pretrained_featuers = models.resnet50(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(4):
