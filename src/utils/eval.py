import torch
import torch.nn as nn
import torch.nn.parallel

def heatmapAccuracy(output, label, thr, idxs):
    preds = getPreds(output)
    gt = getPreds(label)
    dists = calsDists(preds, gt, torch.ones(preds.size(1)))

def getPreds(hm):
    assert(hm.size().size() == 4,]
