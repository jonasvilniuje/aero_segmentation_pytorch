import os
import shutil
from random import sample

def move_validation_data(source_dir, validation_dir, num_files):
    """
    Move a specified number of images and their corresponding masks to a validation directory.
    Images and masks are assumed to have the same basename but different extensions (images in .jpg, masks in .png).

    Args:
    - source_dir (str): Path to the source directory containing 'img' and 'masks' subdirectories.
    - validation_dir (str): Path to the validation directory where 'img' and 'masks' will be stored.
    - num_files (int): Number of files to move to the validation set.
    """
    img_source = os.path.join(source_dir, 'img')
    masks_source = os.path.join(source_dir, 'mask')
    
    img_validation = os.path.join(validation_dir, 'img')
    masks_validation = os.path.join(validation_dir, 'mask')
    
    # Create validation directories if they don't exist
    os.makedirs(img_validation, exist_ok=True)
    os.makedirs(masks_validation, exist_ok=True)
    
    # Get a list of filenames (without considering the file extension for images)
    filenames = os.listdir(img_source)
    filenames = [f for f in filenames if not f.startswith('.')]  # Ignore hidden files
    # Assume image files are .jpg for this example
    selected_files = sample(filenames, num_files)
    
    # Move the selected images and their corresponding masks
    for filename in selected_files:
        base_name = os.path.splitext(filename)[0]
        
        img_src_path = os.path.join(img_source, filename)
        mask_filename = base_name + '.png'  # Change the extension for the mask
        mask_src_path = os.path.join(masks_source, mask_filename)
        
        img_dest_path = os.path.join(img_validation, filename)
        mask_dest_path = os.path.join(masks_validation, mask_filename)
        
        shutil.move(img_src_path, img_dest_path)
        shutil.move(mask_src_path, mask_dest_path)
        
        print(f'Moved {filename} and its mask to validation set.')







# from utils.moveImages import move_validation_data

# # Example usage
# source_dir = 'airbus-vessel-recognition/training_data_40k/training_data'
# validation_dir = 'airbus-vessel-recognition/training_data_40k/validation'
# num_files = 2000  # Number of files to move

# move_validation_data(source_dir, validation_dir, num_files)
