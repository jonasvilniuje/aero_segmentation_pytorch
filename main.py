import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image
import configparser
import time

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1e-6  # Smoothing factor to avoid division by zero
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = torch.sum(input_flat * target_flat)
        union = torch.sum(input_flat) + torch.sum(target_flat)

        dice_coeff = (2. * intersection + smooth) / (union + smooth)

        dice_loss = 1 - dice_coeff
        return dice_loss

# Define U-Net architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()  # Output between 0 and 1 for binary segmentation
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define dataset
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, fixed_size=None):
        super().__init__(root, transform=transform)
        if fixed_size is not None:
            self.samples = self.samples[:fixed_size]  # Limiting samples to a fixed size

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        # Load the corresponding mask
        mask_path = path.replace('img', 'mask')
        mask_path = mask_path.replace('.jpg', '.png')  # Change file extension to PNG
        mask = Image.open(mask_path)
        mask = transforms.Resize((256, 256))(mask)
        mask = transforms.ToTensor()(mask)

        return sample, mask

print("work start...")
print("is cuda available?:", torch.cuda.is_available())
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parse root folder of images and masks
root = "airbus-vessel-recognition/training_data_1k_256"

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

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize U-Net model
model = UNet().to(device)  # Move model to GPU if available

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    start_time = time.time()  # Start timer for epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Move data and target to the same device as model
        data, target = data.to(device), target.to(device)

        output = model(data)

        # Ensure output has the correct shape
        target = target[:, 0, :, :].unsqueeze(1)

        criterion = DiceLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}')

    end_time = time.time()  # End timer for epoch
    epoch_time = end_time - start_time  # Calculate epoch duration
    print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f} seconds')
    
    # Test the model on test images
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            # Move data to the same device as model
            data = data.to(device)

            output = model(data)
            save_image(output, f'segmentation_results/test_segmentation_result_epoch_{epoch+1}_image_{i+1}.png')

print("Segmentation results for test images after each epoch saved.")
