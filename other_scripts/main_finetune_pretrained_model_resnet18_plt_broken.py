import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from torch.utils.data.dataset import random_split
from torchvision.models.resnet import resnet18, ResNet18_Weights

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define your model architecture
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Use a pretrained ResNet18, but remove the final fully connected layer
        weights = ResNet18_Weights.DEFAULT
        self.encoder = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        # Decoder with adjusted upsampling to reach 256x256 output size
        self.decoder = nn.Sequential(
            # Start with 512 channels from the encoder, output size is [batch, 512, 8, 8]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # Output: [batch, 256, 16, 16]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # Output: [batch, 128, 32, 32]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # Output: [batch, 64, 64, 64]
            nn.ReLU(inplace=True),
            # Additional upsampling layers
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # Output: [batch, 64, 128, 128]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: [batch, 32, 256, 256]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1), # Final layer to get the desired channel size
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def dice_loss(pred, target, smooth=1e-6):
    # Resize pred to match the size of target
    pred = torch.nn.functional.interpolate(pred, size=target.size()[2:], mode='bilinear', align_corners=False)
    
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def bce_loss(device):
    class_weight = torch.tensor([0.05, 0.95]).to(device)  # Assuming [background, vessel] as classes
    class_weight = class_weight / class_weight.sum()  # Normalize to sum to 1
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight[1]]))

def weighted_dice_loss(pred, target, weights=[0.05, 0.95], smooth=1e-6):
    pred = F.interpolate(pred, size=target.size()[2:], mode='bilinear', align_corners=False)
    pred = pred.contiguous()
    target = target.contiguous()

    # Calculate dice coefficient for each instance in the batch
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)

    # Calculate weights for each instance in the batch
    w = weights[1] * target.sum(dim=(2, 3)) + weights[0] * (1 - target).sum(dim=(2, 3))

    # Calculate weighted dice loss
    weighted_dice = (1 - dice) * w
    weighted_dice_loss = weighted_dice.sum() / w.sum()
    
    return weighted_dice_loss

def combined_loss(pred, target, bce_weight=0.5, dice_weights=[0.05, 0.95], device="cuda"):
    bce = bce_loss(pred, target)
    dice = weighted_dice_loss(pred, target, weights=dice_weights)
    return bce_weight * bce + (1 - bce_weight) * dice

# Define training function
def train(model, train_loader, criterion, optimizer, device):
    print('entering training loop...')
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        print('for images, masks in train_loader...')
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # Resize model outputs to match target size
        outputs_resized = torch.nn.functional.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
        loss = criterion(outputs_resized, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        print(f'outputs: {np.shape(outputs)}, outputs_resized: {np.shape(outputs_resized)}')
        print(f'running_loss {running_loss:.4f}')

    return running_loss / len(train_loader.dataset)

# Define evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            # Resize model outputs to match target size
            outputs_resized = torch.nn.functional.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
            loss = criterion(outputs_resized, masks)

            running_loss += loss.item() * images.size(0)

            preds.append(outputs_resized.cpu().numpy())
            targets.append(masks.cpu().numpy())

    epoch_loss = running_loss / len(test_loader.dataset)
    return epoch_loss, np.concatenate(preds), np.concatenate(targets)

# Define your main function
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set hyperparameters
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 10

    # Create dataset and dataloaders
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_dataset = CustomDataset(image_dir="airbus-vessel-recognition/training_data_1k_256/train/img/",
                                  mask_dir="airbus-vessel-recognition/training_data_1k_256/train/mask/",
                                  transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_size = int(0.8 * len(train_dataset))  # 80% for training
    val_size = len(train_dataset) - train_size  # Remaining 20% for validation

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    test_dataset = CustomDataset(image_dir="airbus-vessel-recognition/training_data_1k_256/test/img/",
                                mask_dir="airbus-vessel-recognition/training_data_1k_256/test/mask/",
                                transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = CustomModel().to(device)

    # losses:
    # criterion = dice_loss
    criterion = bce_loss(device)
    # criterion = weighted_dice_loss
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []
    val_accuracy = []
    val_dice_coefficient = []

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, preds, targets = evaluate(model, val_loader, criterion, device)
        print(f'train_loss {train_loss}')
        print(f'val_loss {val_loss}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'train_losses {train_losses}')
        print(f'val_losses {val_losses}')

        # Calculate metrics
        preds_binary = (preds > 0.5).astype(np.uint8)
        val_accuracy.append(accuracy_score(targets.flatten(), preds_binary.flatten()))
        val_dice_coefficient.append(f1_score(targets.flatten(), preds_binary.flatten()))

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Plot loss curves
        plt.plot(range(num_epochs), train_losses, label='Train Loss')
        plt.plot(range(num_epochs), val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()
        plt.savefig('segmentation_results/20240319_dice_resnet18/train_val_loss_graph.png')

    # Plot loss curves
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('segmentation_results/20240319_dice_resnet18/train_val_loss_graph.png')

    plt.plot(range(num_epochs), val_accuracy, label='Val Accuracy')
    plt.plot(range(num_epochs), val_dice_coefficient, label='Val Dice coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Val Accuracy and Dice Coefficient')
    plt.legend()
    # plt.show()
    plt.savefig('segmentation_results/20240319_dice_resnet18/val_acc_dice.png')

    test_loss, preds, targets = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')

    # Calculate metrics
    preds_binary = (preds > 0.5).astype(np.uint8)
    accuracy = accuracy_score(targets.flatten(), preds_binary.flatten())
    dice_coefficient = f1_score(targets.flatten(), preds_binary.flatten())

    print(f"Accuracy: {accuracy:.4f}, Dice Coefficient: {dice_coefficient:.4f}")

if __name__ == "__main__":
    main()
