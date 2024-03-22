import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1e-6  # Smoothing factor to avoid division by zero
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = torch.sum(input_flat * target_flat)
        union = torch.sum(input_flat) + torch.sum(target_flat)

        dice_coeff = (2. * intersection + smooth) / (union + smooth)

        dice_loss = 1 - dice_coeff
        return dice_loss
    
def bce_loss(device):
    class_weights = torch.tensor([0.05, 0.95], dtype=torch.float).to(device)
    pos_weight = torch.tensor([class_weights[1]], dtype=torch.float).to(device)  # pos_weight should be a tensor
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
