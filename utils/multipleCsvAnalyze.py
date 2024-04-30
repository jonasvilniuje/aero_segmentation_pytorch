import os
import pandas as pd
import matplotlib.pyplot as plt

def read_iou_data(base_path, folders):
    iou_data = {}
    
    for folder in folders:
        path = os.path.join(base_path, folder, 'val.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Assuming IOU is stored as a string list, convert it to a list of floats
            iou_series = eval(df['iou'].iloc[0])
            iou_data[folder] = iou_series
        else:
            print(f"File not found: {path}")
    
    return iou_data

def plot_iou(iou_data):
    plt.figure(figsize=(10, 6))
    
    for label, ious in iou_data.items():
        epochs = list(range(1, len(ious) + 1))
        plt.plot(epochs, ious, label=label)
    
    plt.title('IOU Metrics Over Epochs (validation)')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.grid(True)
    plt.show()


base_path = 'hpc_results/results/'  # Change this to your base path
folders = [ 'unet_baseline_1000_50E_8B',
            'unet_baseline_1000_50E_16B',
            'unet_baseline_1000_50E_32B',
            'unet_baseline_1000_50E_64B',
            'unet_baseline_1000_50E_128B']  # List the folders you want to include

iou_data = read_iou_data(base_path, folders)
# plot_iou(iou_data)

folders = [ 'unet_baseline_5000_50E_8B',
            'unet_baseline_5000_50E_16B',
            'unet_baseline_5000_50E_32B_depth4',
            'unet_baseline_5000_50E_64B',
            'unet_baseline_5000_50E_128B']  # List the folders you want to include

iou_data = read_iou_data(base_path, folders)
# plot_iou(iou_data)

folders = [ 'unet_baseline_5000_50E_32B_depth4',
            'unet_baseline_1000_50E_32B']  # List the folders you want to include

iou_data = read_iou_data(base_path, folders)
plot_iou(iou_data)