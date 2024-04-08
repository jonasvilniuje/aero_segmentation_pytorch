from torchvision import transforms

def define_transformations():
    return transforms.Compose([
        # Existing transformations
        transforms.Resize((256, 256)),  # Ensure images are resized if needed
        transforms.ToTensor(),

        # Augmentations:
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flips
        transforms.RandomRotation(degrees=45),  # Random rotations
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
    ])