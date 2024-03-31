from PIL import Image
import os

def find_images_with_dimension(folder_path, target_dimension=(1, 256, 256)):
    """
    Find images in a folder that match the target dimension.

    Parameters:
    - folder_path: Path to the folder containing the images.
    - target_dimension: A tuple representing the target dimension (channels, width, height).

    Returns:
    - A list of image file paths that match the target dimension.
    """
    matching_images = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            with Image.open(file_path) as img:
                # PIL.Image does not directly provide channel information
                # Assuming RGB (3 channels) or single channel (grayscale)
                width, height = img.size
                channels = 1 if img.mode == 'L' else 3  # 'L' is the mode for grayscale images

                # Check if the image dimensions match the target
                if (channels, width, height) == target_dimension:
                    matching_images.append(file_path)
                    
        except IOError:
            # If the file cannot be opened as an image, skip it
            print(f"Skipping non-image file: {file_path}")

    return matching_images

# Example usage
folder_path = 'airbus-vessel-recognition/training_data_1k_256/train/img'
target_dimension = (1, 256, 256)  # Specify the target dimension
matching_images = find_images_with_dimension(folder_path, target_dimension)
print("Images matching the target dimension:")
for img_path in matching_images:
    print(img_path)
