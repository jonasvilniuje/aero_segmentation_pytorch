def load_image(image_path):
    import torchvision.transforms as transforms
    from PIL import Image

    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    return image.unsqueeze(0)

