
from PIL import Image

def get_min_max_pixel_values(image_path):
    image = Image.open(image_path)
    pixels = image.getdata()
    min_val = min(pixels)
    max_val = max(pixels)
    return min_val, max_val

def visualize_segmentation(image, ground_truth, segmentation_mask, image_name="output"):
    import matplotlib.pyplot as plt
    import numpy as np

    segmentation_mask = (segmentation_mask > 0.5).float() # apply threshold

    plt.figure(figsize=(12, 4))
    # plt.subplots(1, 3, figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.transpose(ground_truth.numpy(), (1, 2, 0)))
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(np.transpose(segmentation_mask.detach().numpy(), (1, 2, 0)))
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.savefig(f'results\\deeplabv3_resnet50\\{image_name}.png')
    plt.close()

