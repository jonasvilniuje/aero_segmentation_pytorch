def calculate_iou(output, target):
    smooth = 1e-6  # To avoid division by zero
    
    # Threshold predictions to create a binary mask
    output_bin = (output > 0.5).float()
    
    # Ensure target is a binary mask (assuming target is already [0,1])
    target_bin = target.float()
    
    # Calculate intersection and union
    intersection = (output_bin * target_bin).sum((1, 2, 3))  # Element-wise multiplication and sum
    union = output_bin.sum((1, 2, 3)) + target_bin.sum((1, 2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    # Return the mean IoU for the batch
    return iou.mean()