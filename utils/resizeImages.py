from PIL import Image
import os

def resize_image(input_path, output_path, new_size):
    """
    Resize an image file and save it to the output path.

    Args:
    input_path (str): Path to the input image file.
    output_path (str): Path to save the resized image.
    new_size (tuple): Desired size of the output image in the format (width, height).
    """
    try:
        with Image.open(input_path) as img:
            img_resized = img.resize(new_size, Image.LANCZOS)
            img_resized.save(output_path)
        print(f"Image resized successfully and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def batch_resize(input_folder, output_folder, new_size):
    """
    Batch resize all image files in a folder.

    Args:
    input_folder (str): Path to the folder containing input image files.
    output_folder (str): Path to save the resized images.
    new_size (tuple): Desired size of the output images in the format (width, height).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image(input_path, output_path, new_size)

# Example usage
if __name__ == "__main__":
    input_folder = "airbus-vessel-recognition/testing_data_768/mask"
    output_folder = "airbus-vessel-recognition/testing_data/mask"
    new_size = (256, 256)  # Desired size (width, height)

    batch_resize(input_folder, output_folder, new_size)
