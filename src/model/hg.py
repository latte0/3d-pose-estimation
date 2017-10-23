import torch
import torch.nn as nn
import torch.nn.parallel

import residual

def hourglass(n, f, opt, model):

    for i in range(1,opt.nModules):
        model = Residual(f,f,model)

    low1 = model.add(nn.SpatialMaxPooling(2,2,2,2))

    for i in range(1,opt.nModules)

def createModel(model):

    model  = nn.Sequential()


