import numpy as np

def calculate_m_iou_and_dice(output, target):
    smooth = 1e-6  # To avoid division by zero
    
    # Threshold predictions to create a binary mask
    output_bin = (output > 0.5).float()
    
    # Ensure target is a binary mask (assuming target is already [0,1])
    target_bin = target.float()
    
    # Calculate intersection and union
    intersection = (output_bin * target_bin).sum((1, 2, 3))  # Element-wise multiplication and sum
    union = output_bin.sum((1, 2, 3)) + target_bin.sum((1, 2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    # Assuming intersection and union are PyTorch tensors
    dice = 2. * intersection / union
    dice[union == 0] = 1.0  # Set dice to 1.0 where union is zero

    # Return the mean IoU and mean Dice for the batch
    return iou.mean(), dice.mean()

def calculate_metrics(output, target):
    m_iou, m_dice = calculate_m_iou_and_dice(output, target)

    # Threshold predictions to create a binary mask
    output_bin = (output > 0.5).float()
    
    # Ensure target is a binary mask (assuming target is already [0,1])
    target_bin = target.float()

    TP = ((output_bin == 1) & (target_bin == 1)).sum()
    FP = ((output_bin == 1) & (target_bin == 0)).sum()
    FN = ((output_bin == 0) & (target_bin == 1)).sum()
    # TN = (output_bin == 0) & (target_bin == 0).sum()  # TN is not needed for these metrics
    
    print(f'TP {TP}, FP {FP}, FN {FN}')
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return m_iou.item(), m_dice.item(), precision.item(), recall.item(), f1_score

def print_metrics(m_iou, m_dice, precision, recall, f1_score):
    r = 4 # the decimal point to round to
    print(f'm_iou: {round(m_iou, r)}, dice: {round(m_dice, r)}, precision: {round(precision, r)}, recall: {round(recall, r)}, f1_score: {f1_score}'
    )
    
# def dice():
