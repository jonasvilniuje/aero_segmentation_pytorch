import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image
import sys

# parse root folder of images and masks
root = "airbus-vessel-recognition/training_data_1k_256/train"

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
    def __init__(self, root, transform=None, fixed_size=100):
        super().__init__(root, transform=transform)
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

fixed_size = 100  # Fixed number of samples
dataset = CustomImageFolder(root, transform=transform, fixed_size=fixed_size)

# Define data loader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize U-Net model
model = UNet()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)

        # Ensure output has the correct shape
        # output = output.squeeze(1)  # Remove the channel dimension

        # Resize the target tensor to match the output shape
        target = target[:, 0, :, :]  # Keep only the first channel (assuming it represents the mask)
        target = target.unsqueeze(1)  # Add the channel dimension back
        target = target.expand(-1, output.size(1), -1, -1)  # Resize the target to match the output size

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item()}')

# Save example result
example_input = next(iter(data_loader))[0]
example_output = model(example_input)
save_image(example_output, 'segmentation_result.png')
