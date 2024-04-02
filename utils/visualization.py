from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import configparser
import os

def create_folder_for_results():
    config = configparser.ConfigParser()
    config.read('env.config')

    model_name = config['Model']['name']
    fixed_train_size = int(config['Model']['fixed_train_size'])
    num_epochs = config['Model']['num_epochs']
    batch_size = config['Model']['batch_size']
    save_path = f'results/{model_name}_{fixed_train_size}_{num_epochs}E_{batch_size}B'

    os.makedirs(save_path, exist_ok=True)

    return save_path

def get_min_max_pixel_values(image_path):
    image = Image.open(image_path)
    pixels = image.getdata()
    min_val = min(pixels)
    max_val = max(pixels)
    return min_val, max_val

def visualize_segmentation(image, ground_truth, segmentation_mask, image_name="output"):
    save_path = create_folder_for_results()
    segmentation_mask = (segmentation_mask > 0.5).float() # apply threshold

    plt.figure(figsize=(12, 4))
    # plt.subplots(1, 3, figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.transpose(ground_truth.cpu().numpy(), (1, 2, 0)))
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(np.transpose(segmentation_mask.detach().cpu().numpy(), (1, 2, 0)))
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.savefig(f'{save_path}/{image_name}.png')
    plt.close()

def plot_metrics(metrics, metric_name):
    save_path = create_folder_for_results()
    
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train'][metric_name], label=f'Training {metric_name}')
    plt.plot(metrics['val'][metric_name], label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    # plt.show()
    
    plt.savefig(f'{save_path}/{metric_name}.png')
