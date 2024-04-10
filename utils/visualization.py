from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import configparser
import os

def create_folder_for_results(folder_name=None):
    config = configparser.ConfigParser()
    config.read('env.config')

    model_name = config['Model']['name']
    fixed_train_size = int(config['Model']['fixed_train_size'])
    num_epochs = config['Model']['num_epochs']
    batch_size = config['Model']['batch_size']

    if folder_name is None:
        folder_name = f'{model_name}_{fixed_train_size}_{num_epochs}E_{batch_size}B'
    
    save_path = f'results/{folder_name}'

    os.makedirs(save_path, exist_ok=True)

    return save_path

def get_min_max_pixel_values(image_path):
    image = Image.open(image_path)
    pixels = image.getdata()
    min_val = min(pixels)
    max_val = max(pixels)
    return min_val, max_val

def visualize_segmentation(image, ground_truth, segmentation_mask, image_name="output", show_only=False):
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

    if show_only:
        plt.show()
    else:
        plt.savefig(f'{save_path}/{image_name}.png')
    plt.close()

def plot_metrics(metrics, metric_name, show_only=False):
    save_path = create_folder_for_results()
    
    plt.figure(figsize=(10, 5))
    train_metrics = metrics['train'][metric_name]
    val_metrics = metrics['val'][metric_name]
    
    plt.plot(train_metrics, label=f'Training {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    highest_train = max(train_metrics)
    highest_val = max(val_metrics)
    highest_train_epoch = train_metrics.index(highest_train)
    highest_val_epoch = val_metrics.index(highest_val)
    
    plt.scatter(highest_train_epoch, highest_train, color='blue', s=50, zorder=5)
    plt.scatter(highest_val_epoch, highest_val, color='orange', s=50, zorder=5)
    plt.text(highest_train_epoch, highest_train, f' {highest_train:.2f}', verticalalignment='bottom')
    plt.text(highest_val_epoch, highest_val, f' {highest_val:.2f}', verticalalignment='bottom')
    
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    
    if show_only:
        plt.show()
    else:
        # Save the plot to a file only if not showing it directly
        plt.savefig(f'{save_path}/{metric_name}.png')
    
    plt.close()  # Correctly close the plot after saving or showing
    
def print_model_parameters(model):
    print('model name:', model.name)
    # Counting parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    # Counting all parameters, including those not requiring gradients
    total_all_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters (including non-trainable): {total_all_params}")
