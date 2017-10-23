import torch
import torch.nn as nn
import torch.nn.parallel

conv = nn.SpatialConvolutino
batchnorm = nn.SpatialBatchNormalization
relu = nn.ReLU


def convBlock(numIn, numOut, model):
    return model.add(batchnorm(numIn))
            .add(relu(true))
            .add(conv(numIn,numOut/2,1,1))
            .add(batchnorm(numOut/2))
            .add(relu(true))
            .add(conv(numOut/2, numOut/2,3,3,1,1,1,1))
            .add(batchnorm(numOUt/2))
            .add(relu(true))
            .add(conv(numOut/2,numOut,1,1))


def skipLayer(numIn,numOut,model):
    if numIn == numOut:
        return nn.Identity()
    else
        return model.add(conv(numIn,numOut,1,1))


#def Residual(numIn, numOut, model):
#    return model.add(nn.ConcatTable().add(convBlock(numIn,numO)))
