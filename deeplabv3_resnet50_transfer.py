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
from utils.visualization import visualize_segmentation
from utils.metrics import calculate_metrics, print_metrics

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

    # Fixed number of samples for training and testing
    fixed_train_size = 128
    fixed_test_size = 16
    batch_size = 8

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

def process_output(output, threshold=0.5):
        # Threshold predictions to create a binary mask
    output_bin = (output > 0.5).float()

    return output_bin

def init_deeplabv3_resnet50_model(device):
    # Initialize DeepLabv3 model
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet50(weights=weights).to(device)

    # Modify the output layer for binary segmentation
    num_classes = 1  # Binary segmentation
    in_features = model.classifier[-1].in_channels

    print(f'in_features: {model.classifier[-1].in_channels}')
    model.classifier[-1] = nn.Conv2d(in_features, num_classes, kernel_size=1)
    
    return model


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device) # ground truth imgs and masks
        
        optimizer.zero_grad()
        outputs = model(images)['out'] # predictions

        m_iou, dice, precision, recall, f1_score = calculate_metrics(outputs, masks)
        # outputs = process_output(outputs) # apply threshold
        print_metrics(m_iou, dice, precision, recall, f1_score )


        for i in range(0, outputs.shape[0]):
            # print(i)
            visualize_segmentation(images[i], masks[i], outputs[i], f'img_{i}')

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    return total_loss / len(train_loader.dataset)

def main():
    # Check if GPU is available
    print("is cuda available?:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = init_data()
    model = init_deeplabv3_resnet50_model(device)
    
    background_percentage = 99
    target_percentage = 1

    # Calculate pos_weight
    pos_weight_value = background_percentage / target_percentage
    pos_weight = torch.tensor([pos_weight_value])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    model.eval()
if __name__ == "__main__":
    main()
