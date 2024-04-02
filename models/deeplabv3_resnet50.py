import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def init_deeplabv3_resnet50_model(device):
    # Initialize DeepLabv3 model
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet50(weights=weights).to(device)

    # Modify the output layer for binary segmentation
    num_classes = 1  # Binary segmentation
    in_features = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_features, num_classes, kernel_size=1)
    
    return model