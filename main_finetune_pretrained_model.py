import torch
import torch.nn as nn
import torch.optim as optim
import configparser
import time
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor
from utils.dataLoading import CustomImageFolder
from utils.lossFunctions import DiceLoss

# Function to convert target tensor to one-hot encoded format
def one_hot_encode_target(target, num_classes):
    batch_size, channels, height, width = target.size()
    target_one_hot = torch.zeros(batch_size, num_classes, height, width, dtype=torch.float32)
    for class_idx in range(num_classes):
        target_one_hot[:, class_idx, :, :] = (target[:, 0, :, :] == class_idx).float()
    return target_one_hot


# 1. Prepare Your Dataset
# Define transformations
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
])

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read train_root from env.config file
config = configparser.ConfigParser()
config.read('env.config')
train_root = config['Paths']['train_root']
test_root = config['Paths']['test_root']

# Fixed number of samples for training and testing
fixed_train_size = 100
fixed_test_size = 10

# Define data loaders for training and testing
train_dataset = CustomImageFolder(train_root, transform=transform, fixed_size=fixed_train_size)
test_dataset = CustomImageFolder(test_root, transform=transform, fixed_size=fixed_test_size)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# 3. Initialize Pre-trained Model
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(pretrained=True, weights=weights).to(device)

# 4. Modify the Final Layer
num_classes = 2  # Adjust based on your dataset
model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

# 5. Define Loss Function and Optimizer
# criterion = DiceLoss()  # Use DiceLoss for multi-class segmentation
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop

num_epochs = 10
for epoch in range(num_epochs):
    start_time = time.time()  # Start timer for epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Move data and target to the same device as model
        data, target = data.to(device), target.to(device)

        output = model(data)['out']

        # One-hot encode the target tensor
        target_one_hot = one_hot_encode_target(target, num_classes)

        loss = criterion(output, target_one_hot)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}')
    
    end_time = time.time()  # End timer for epoch
    epoch_time = end_time - start_time  # Calculate epoch duration
    print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f} seconds')
    
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(val_loader):
            # Move data to the same device as model
            data = data.to(device)

            output = model(data)
            image_name = os.path.basename(test_dataset.samples[i][0].split('/')[-1])
            image_name = os.path.splitext(image_name)[0]
            print(image_name, os.path.basename(test_dataset.samples[i][0].split('/')[-1]), test_dataset.samples[i][0].split('/')[-1])
            save_image(output, f'segmentation_results/result_epoch_{epoch+1}_image_{i+1}_{image_name}.png')

    print("Segmentation results for test images after each epoch saved.")
# 7. Evaluate Performance (Optional)
# Evaluate on test set using appropriate metrics

# 8. Adjust Hyperparameters and Repeat as Necessary
