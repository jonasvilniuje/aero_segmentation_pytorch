import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import configparser
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from utils.dataLoading import CustomImageFolder
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
# from utils.lossFunctions import dice_loss
# from utils.metrics import calculate_iou  # Assuming you have defined calculate_iou
from utils.visualization import visualize_segmentation, plot_metrics
# from utils.metrics import calculate_metrics, print_metrics

# Define transformations
transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Read train_root from env.config file
config = configparser.ConfigParser()
config.read('env.config')

def init_data():
    train_root = config['Paths']['train_root']
    val_root = config['Paths']['val_root']
    test_root = config['Paths']['test_root']
    fixed_train_size = int(config['Model']['fixed_train_size'])
    fixed_valid_size = int(config['Model']['fixed_valid_size'])
    fixed_test_size = int(config['Model']['fixed_test_size'])
    batch_size = int(config['Model']['batch_size'])

    if not torch.cuda.is_available():
        fixed_train_size = 128
        fixed_valid_size = 16
        fixed_test_size = 16

    # Define data loaders for training and testing
    train_dataset = CustomImageFolder(train_root, transform=transform, fixed_size=fixed_train_size)
    val_dataset = CustomImageFolder(val_root, transform=transform, fixed_size=fixed_valid_size)
    test_dataset = CustomImageFolder(test_root, transform=transform, fixed_size=fixed_test_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Contracting Path (Encoder)
        self.enc_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Expansive Path (Decoder)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
        # Up-sampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc_conv1(x))
        x2 = self.pool1(x1)
        x3 = F.relu(self.enc_conv2(x2))
        
        # Decoder
        x4 = self.upsample(x3)
        x5 = F.relu(self.dec_conv1(x4))
        x6 = self.dec_conv2(x5)
        
        return x6

def init_unet_model(device):
    model = UNet(in_channels=3, out_channels=1)
    model.to(device)
    
    # Counting parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    # Counting all parameters, including those not requiring gradients
    total_all_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters (including non-trainable): {total_all_params}")

    return model

def init_deeplabv3_resnet50_model(device):
    # Initialize DeepLabv3 model
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet50(weights=weights).to(device)

    # Modify the output layer for binary segmentation
    num_classes = 1  # Binary segmentation
    in_features = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_features, num_classes, kernel_size=1)
    
    return model

def loop(model, loader, criterion, optimizer, device, phase="training"):
    if phase == "training":
        model.train()
    else:
        model.eval()
    
    # Initialize for IoU calculation
    total_TP, total_FP, total_FN, iou = 0, 0, 0, 0
    total_loss = 0.0

    with torch.set_grad_enabled(phase == "training"):
        for images, masks in loader:
            # Model prediction and any necessary processing here
            images, masks = images.to(device), masks.to(device)

            # Handle different model output types
            model_output = model(images)
            outputs = model_output if isinstance(model_output, torch.Tensor) else model_output['out']
            
            loss = criterion(outputs, masks)

            if phase == "training":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            # Convert outputs and masks to binary if necessary, e.g., for segmentation tasks
            output_bin = (outputs > 0.5).float()
            mask_bin = masks.float()  # Assuming masks are already binary
            
            # Calculate TP, FP, FN (and TN if needed) for the current batch
            TP = ((output_bin == 1) & (mask_bin == 1)).sum().item()
            FP = ((output_bin == 1) & (mask_bin == 0)).sum().item()
            FN = ((output_bin == 0) & (mask_bin == 1)).sum().item()
            
            # Accumulate metrics components
            total_TP += TP
            total_FP += FP
            total_FN += FN

            if phase == "testing":
                print(phase)
                # iterate through imgs, masks and outputs to plot them
                for i in range(0, len(outputs)):
                    visualize_segmentation(images[i], masks[i], outputs[i], image_name=f"output{i}_")
        
        # Calculate metrics using the accumulated values
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
    
    avg_loss = total_loss / len(loader.dataset)

    return {
        'iou': iou,
        'avg_loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def main():
    # Check if GPU is available
    print("is cuda available?:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = config['Model']['name']

    train_loader, val_loader, test_loader = init_data()
    if model_name == 'unet':
        model = init_unet_model(device)
    elif model_name == 'deeplabv3_resnet50':
        model = init_deeplabv3_resnet50_model(device)
    else:
        model = init_unet_model(device)
        
    model.to(device)
    
    background_percentage = 99
    target_percentage = 1
    # Calculate pos_weight
    pos_weight_value = background_percentage / target_percentage
    pos_weight = torch.tensor([pos_weight_value])
    pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    metrics = {
        'train': {'avg_loss': [], 'iou': [], 'precision': [], 'recall': [], 'f1_score': []}, # accuracy is missing
        'val': {'avg_loss': [], 'iou': [], 'precision': [], 'recall': [], 'f1_score': []}
    }

    num_epochs = 10
    for epoch in range(num_epochs):
        # Training phase
        train_metrics = loop(model, train_loader, criterion, optimizer, device, phase="training")
        for key in metrics['train'].keys():
            metrics['train'][key].append(train_metrics[key])
        print(f"Training: {train_metrics}")
    
        # Validation phase
        val_metrics = loop(model, val_loader, criterion, None, device, phase="validation")
        for key in metrics['val'].keys():
            metrics['val'][key].append(val_metrics[key])
        print(f"Validation: {val_metrics}")

        print(f"Epoch {epoch+1}/{num_epochs}")

    for metric_name in metrics['train'].keys(): 
        plot_metrics(metrics, metric_name) # tekes care of plotting val metrics as well
    
    # Test the model
    test_metrics = loop(model, test_loader, criterion, None, device, phase="testing")
    print(f'test_metrics: {test_metrics}')

if __name__ == "__main__":
    main()

