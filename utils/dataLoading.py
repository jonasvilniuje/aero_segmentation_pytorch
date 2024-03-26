from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    return image.unsqueeze(0)

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
        mask = Image.open(mask_path).convert("L")
        mask = transforms.Resize((256, 256))(mask)
        mask = transforms.ToTensor()(mask)

        return sample, mask