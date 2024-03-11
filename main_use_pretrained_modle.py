import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

class SegmentationModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(SegmentationModel, self).__init__()
        # self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        # self.model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        # self.model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.classifier[-1] = torch.nn.Conv2d(512, num_classes, kernel_size=(2, 2), stride=1)

    def forward(self, x):
        return self.model(x)['out']

def load_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    return image.unsqueeze(0)

def segment_image(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        predicted_labels = torch.argmax(output, dim=1)
    return predicted_labels.squeeze(0)

def visualize_segmentation(image, segmentation_mask):
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_mask)
    plt.title("Segmentation Mask")
    plt.axis('off')

    # plt.savefig('/c/Users/Master/Documents/VU/ML/aero_pytorch/001aee007.png')
    plt.savefig(f'C:\\Users\\Master\\Documents\\VU\\ML\\aero_pytorch\\{image_name}.png')
    print('4')
    plt.close()

def get_min_max_pixel_values(image_path):
    image = Image.open(image_path)
    pixels = image.getdata()
    min_val = min(pixels)
    max_val = max(pixels)
    return min_val, max_val


if __name__ == "__main__":
    # Define paths and parameters
    image_name = "0006c52e8"
    image_path = f'airbus-vessel-recognition/training_data_1k_256/test/img/{image_name}.jpg'
    num_classes = 2  # Number of classes for DeepLabv3
    print('0')
    # Load the model
    model = SegmentationModel(num_classes)
    print('1')
    # Load and preprocess the image
    image = load_image(image_path)
    print('2')
    # Perform segmentation
    segmentation_mask = segment_image(model, image)
    print('3')
    # Visualize the result
    visualize_segmentation(image.squeeze(), segmentation_mask)
    print('5')

    min_val, max_val = get_min_max_pixel_values(image_path)
    print("Minimum pixel value:", min_val)
    print("Maximum pixel value:", max_val)