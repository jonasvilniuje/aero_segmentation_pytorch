import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import configparser
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from utils.dataLoading import CustomImageFolder  # Assuming you have custom data loading utilities
# from utils.lossFunctions import dice_loss  # Assuming you have defined dice_loss
# from utils.metrics import calculate_iou  # Assuming you have defined calculate_iou

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    return total_loss / len(train_loader.dataset)

def main():
    print("is cuda available?:", torch.cuda.is_available())
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read train_root from env.config file
    config = configparser.ConfigParser()
    config.read('env.config')
    train_root = config['Paths']['train_root']
    test_root = config['Paths']['test_root']

    # Fixed number of samples for training and testing
    fixed_train_size = 500
    fixed_test_size = 50

    # Define data loaders for training and testing
    train_dataset = CustomImageFolder(train_root, transform=transform, fixed_size=fixed_train_size)
    test_dataset = CustomImageFolder(test_root, transform=transform, fixed_size=fixed_test_size)

    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False)

    # Initialize DeepLabv3 model
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet50(weights=weights).to(device)

    pos_weight = torch.tensor([2.0])  # Adjust based on your dataset
    if torch.cuda.is_available():
        pos_weight = pos_weight.to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

if __name__ == "__main__":
    main()
