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
# from utils.lossFunctions import dice_loss
# from utils.metrics import calculate_iou  # Assuming you have defined calculate_iou
from utils.visualization import visualize_segmentation, plot_metrics
# from utils.metrics import calculate_metrics, print_metrics

# Define transformations
transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def init_data():
    # Read train_root from env.config file
    config = configparser.ConfigParser()
    config.read('env.config')
    train_root = config['Paths']['train_root']
    test_root = config['Paths']['test_root']
    fixed_train_size = int(config['Paths']['fixed_train_size'])
    fixed_test_size = int(config['Paths']['fixed_test_size'])
    batch_size = 8

    if not torch.cuda.is_available():
        fixed_train_size = 128
        fixed_test_size = 16

    # Define data loaders for training and testing
    train_dataset = CustomImageFolder(train_root, transform=transform, fixed_size=fixed_train_size)

    train_size = int(0.8 * len(train_dataset))  # 80% for training
    val_size = len(train_dataset) - train_size  # Remaining 20% for validation

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    test_dataset = CustomImageFolder(test_root, transform=transform, fixed_size=fixed_test_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

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
            outputs = model(images)['out']
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
                    print(f"output{i}")
                    visualize_segmentation(images[i], masks[i], outputs[i], image_name=f"output{i}")
        
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

    train_loader, val_loader, test_loader = init_data()
    model = init_deeplabv3_resnet50_model(device)
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
