import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import plotly.express as px
# from plotly import express as px

def plot_iou_vs_batch_size(csv_file):
    # Load the data from CSV
    data = pd.read_csv(csv_file)
    
    # Prepare the figure and axes for plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique models for legend
    models = data['model_name'].unique()
    
    # Colors for different models
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # Ensure enough colors are present
    if len(models) > len(colors):
        colors = colors * (len(models) // len(colors) + 1)
    
    # Group by model_name and then by batch_size
    grouped = data.groupby('model_name')
    
    # Variable to offset bars for clarity
    bar_width = 2
    idx = 0
    
    # Loop through each group
    for name, group in grouped:
        # Get average IoU for each batch size for the current model
        avg_iou_per_batch = group.groupby('batch_size')['iou'].mean()
        
        # Create bar for each batch size
        ax.bar(avg_iou_per_batch.index + idx * bar_width, avg_iou_per_batch.values, 
               color=colors[idx], width=bar_width, label=name)
        idx += 1
    
    # Adding labels and title
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Average IoU')
    ax.set_title('IoU vs Batch Size Grouped by Model')
    ax.legend(title='Model Name')
    
    # Setting x-ticks to be at the center of the groups of bars
    ax.set_xticks([x + bar_width * (len(models) - 1) / 2 for x in avg_iou_per_batch.index])
    ax.set_xticklabels(avg_iou_per_batch.index)
    
    # Show the plot
    plt.show()

def plot_time_batch_iou(csv_file):
    data = pd.read_csv(csv_file)
    models = data['model_name'].unique()
    batch_sizes = data['batch_size'].unique()
    
    # Create a plot for each model
    for model in models:
        model_data = data[data['model_name'] == model]
        plt.figure(figsize=(10, 6))
        
        for i, batch_size in enumerate(batch_sizes):
            # Filter data for each batch size
            batch_data = model_data[model_data['batch_size'] == batch_size]
            plt.bar(batch_data['time'], batch_data['iou'], width=0.4, label=f'Batch Size {batch_size}')

        plt.title(f'IoU vs Time for {model}')
        plt.xlabel('Time')
        plt.ylabel('IoU')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

def scatter_plot_iou_vs_batch_size(path_to_data):
    # Function to convert hh:mm to total minutes
    def convert_time_to_minutes(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes

    # Load the dataset
    data = pd.read_csv(path_to_data)

    # Convert training time from hh:mm to minutes
    data['training_time_minutes'] = data['training_time'].apply(convert_time_to_minutes)

    # Prepare the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    # Plot 1: Batch Size vs Training Time
    axs[0].scatter(data['batch_size'], data['training_time_minutes'], color='blue', alpha=0.7)
    axs[0].set_title('Batch Size vs Training Time')
    axs[0].set_xlabel('Batch Size')
    axs[0].set_ylabel('Training Time (in minutes)')
    axs[0].grid(True)

    # Adding a trend line to the first plot
    coefficients = np.polyfit(data['batch_size'], data['training_time_minutes'], 1)
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(min(data['batch_size']), max(data['batch_size']), 100)
    y_axis = polynomial(x_axis)
    axs[0].plot(x_axis, y_axis, color='red', linestyle='--')

    # Plot 2: Batch Size vs IOU
    axs[1].scatter(data['batch_size'], data['iou'], color='green', alpha=0.7)
    axs[1].set_title('Batch Size vs IOU')
    axs[1].set_xlabel('Batch Size')
    axs[1].set_ylabel('IOU')
    axs[1].grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()

# Example usage
# plot_time_batch_iou('hpc_results/results/global_test_metrics.csv')
# plot_iou_vs_batch_size('hpc_results/results/global_test_metrics.csv')

scatter_plot_iou_vs_batch_size('hpc_results/results/global_test_metrics.csv')


