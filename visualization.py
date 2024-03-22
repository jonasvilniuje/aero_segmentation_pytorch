
from PIL import Image

def get_min_max_pixel_values(image_path):
    image = Image.open(image_path)
    pixels = image.getdata()
    min_val = min(pixels)
    max_val = max(pixels)
    return min_val, max_val

def visualize_segmentation(image, segmentation_mask, image_name):
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
    plt.savefig(f'C:\\Users\\Master\\Documents\\VU\\ML\\aero_pytorch\\results\\{image_name}.png')
    plt.close()

