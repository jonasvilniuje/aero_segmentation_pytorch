
from PIL import Image

def get_min_max_pixel_values(image_path):
    image = Image.open(image_path)
    pixels = image.getdata()
    min_val = min(pixels)
    max_val = max(pixels)
    return min_val, max_val

# image_path = "your_image.png"
image_path="segmentation_results/test_segmentation_result_epoch_10_image_10.png"
min_val, max_val = get_min_max_pixel_values(image_path)
print("Minimum pixel value:", min_val)
print("Maximum pixel value:", max_val)
