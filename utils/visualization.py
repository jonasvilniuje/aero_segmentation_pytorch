from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv
import configparser
import os

def get_folder_name():
    config = configparser.ConfigParser()
    config.read('env.config')

    model_name = config['Model']['name']
    fixed_train_size = int(config['Model']['fixed_train_size'])
    num_epochs = config['Model']['num_epochs']
    batch_size = config['Model']['batch_size']
    
    return f'{model_name}_{fixed_train_size}_{num_epochs}E_{batch_size}B'

def create_folder_for_results(folder_name=None):
    config = configparser.ConfigParser()
    config.read('env.config')

    if folder_name is None:
        folder_name = get_folder_name()
    
    save_path = f'results/{folder_name}'

    os.makedirs(save_path, exist_ok=True)

    return save_path
    
global batch_visualization_counter
batch_visualization_counter = 0

def visualize_batch(images, ground_truths, segmentation_masks, save_path=".", img_limit=5, batch_limit=3):
    """Visualize images, ground truths, and masks for a batch."""
    global batch_visualization_counter
    batch_visualization_counter += 1
    if batch_visualization_counter > batch_limit: # compile only specified amount of images
        return
    # Limit the number of images to visualize per batch
    num_images = min(images.shape[0], img_limit)
    rows = num_images * 3  # 3 images per item: original, ground truth, predicted mask
    
    plt.figure(figsize=(12, rows * 2))
    
    for i in range(num_images):
        # Original Image
        plt.subplot(num_images, 3, i*3 + 1)
        plt.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
        plt.title(f"Original Image {i+1}")
        plt.axis('off')

        # Ground Truth
        plt.subplot(num_images, 3, i*3 + 2)
        plt.imshow(ground_truths[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title(f"Ground Truth {i+1}")
        plt.axis('off')

        # Predicted Mask
        plt.subplot(num_images, 3, i*3 + 3)
        predicted_mask = (segmentation_masks[i] > 0.5).float()
        plt.imshow(predicted_mask.detach().cpu().numpy().squeeze())
        plt.title(f"Predicted Mask {i+1}")
        plt.axis('off')

    # Save the figure with a unique filename based on the batch counter
    filename = os.path.join(save_path, f"batch_{batch_visualization_counter}_visualization.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization for batch {batch_visualization_counter} to {filename}")


def plot_metrics(metrics, metric_name, save_path, show_only=False):
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

    lowest_train = min(train_metrics)
    lowest_val = min(val_metrics)
    lowest_train_epoch = train_metrics.index(lowest_train)
    lowest_val_epoch = val_metrics.index(lowest_val)

    plt.scatter(lowest_train_epoch, lowest_train, color='green', s=50, zorder=5, marker='v')  # Using a different marker for visual distinction
    plt.scatter(lowest_val_epoch, lowest_val, color='red', s=50, zorder=5, marker='v')
    plt.text(lowest_train_epoch, lowest_train, f' {lowest_train:.2f}', verticalalignment='top')
    plt.text(lowest_val_epoch, lowest_val, f' {lowest_val:.2f}', verticalalignment='top')
    
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

def save_results_to_csv(test_metrics_list, folder_name=None, filename='results'):
    if folder_name is None:
        folder_name = get_folder_name()
        full_path = f'results/{folder_name}/{filename}.csv'

    full_path = f'{folder_name}/{filename}.csv'

    with open(full_path, mode='w', newline='') as file:
        headers = test_metrics_list.keys()
        writer = csv.DictWriter(file, fieldnames=headers)

        writer.writeheader()  
        writer.writerow(test_metrics_list)  # Write the single dictionary

    print(f"Data saved to {full_path}")


def append_results_to_csv(test_metrics, filename='results/global_test_metrics.csv'):
    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        # Extract keys from the dictionary as headers if the file does not exist
        headers = test_metrics.keys()
        writer = csv.DictWriter(file, fieldnames=headers)
        
        # Write headers only if the file does not exist
        if not file_exists:
            writer.writeheader()
        
        # Write the dictionary content
        writer.writerow(test_metrics)

    print(f"Data appended to {filename}")
