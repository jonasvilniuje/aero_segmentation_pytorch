import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt
import numpy as np

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        self.transform = transform

        # Extract image names
        self.image_names = [os.path.basename(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image_name = self.image_names[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, image_name

# Define the FCN model
class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformation to convert PIL images to tensors
transform = Compose([ToTensor()])

# Define dataset and dataloader
image_dir = 'airbus-vessel-recognition/training_data_1k_256/train/img/'
mask_dir = 'airbus-vessel-recognition/training_data_1k_256/train/mask/'
dataset = CustomDataset(image_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleFCN().to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, masks, image_names in dataloader:
        images = images.to(device)  # Move images to device
        masks = masks.to(device)  # Move masks to device
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Save segmentation results for a batch of images
    with torch.no_grad():
        model.eval()
        images, masks, image_names = next(iter(dataloader))
        images = images.to(device)  # Move images to device
        outputs = model(images)

        for i in range(len(images)):
            image_name = image_names[i]
            original_image = images[i].permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
            ground_truth_mask = masks[i][0].cpu().numpy()  # Convert tensor to numpy array
            predicted_mask = outputs[i][0].cpu().numpy()  # Convert tensor to numpy array

            # Save images and masks
            plt.figure(figsize=(12, 6))

            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title('Original Image')
            plt.axis('off')

            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(ground_truth_mask, cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')

            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_mask, cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            # Save plot with image name included
            plt.savefig(f'segmentation_results/pretrained/results_epoch_{epoch+1}_{image_name}.png')
            plt.close()

print('Training finished.')
