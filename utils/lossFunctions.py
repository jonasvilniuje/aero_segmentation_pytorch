import torch
import torch.nn as nn
import torch.nn.functional as F


def dice(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)
    intersection = (inputs * targets).sum()
    dice_coef = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice_coef

def focal(inputs, targets, alpha=0.8, gamma=2):
    inputs = torch.sigmoid(inputs)
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    targets = targets.type(torch.float32)
    at = alpha * targets + (1 - alpha) * (1 - targets)
    pt = (inputs * targets) + ((1 - inputs) * (1 - targets))
    F_loss = at * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()

def bce_loss(device):
    class_weights = torch.tensor([0.05, 0.95], dtype=torch.float).to(device)
    pos_weight = torch.tensor([class_weights[1]], dtype=torch.float).to(device)  # pos_weight should be a tensor
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
