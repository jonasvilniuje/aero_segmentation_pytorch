import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import configparser
import time
from utils.dataLoading import CustomImageFolder
from utils.visualization import visualize_segmentation, plot_metrics, create_folder_for_results, print_model_parameters
from models.unet import init_unet_model
from models.unet_colab import init_unet_model_colab
from models.efficientUnet import init_efficientUNet_model
from models.deeplabv3_resnet50 import init_deeplabv3_resnet50_model
import segmentation_models_pytorch as smp
train_transform = transforms.Compose([
    # Existing transformations
    transforms.Resize((256, 256)),  # Ensure images are resized if needed
    transforms.ToTensor(),

    # Augmentations:
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flips
    transforms.RandomRotation(degrees=45),  # Random rotations
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
])
eval_transform = transforms.Compose([
    transforms.ToTensor()
])

# Read train_root from env.config file
config = configparser.ConfigParser()
config.read('env.config')
train_root = config['Paths']['train_root']
val_root = config['Paths']['val_root']
test_root = config['Paths']['test_root']
fixed_train_size = int(config['Model']['fixed_train_size'])
fixed_valid_size = int(config['Model']['fixed_valid_size'])
fixed_test_size = int(config['Model']['fixed_test_size'])
batch_size = int(config['Model']['batch_size'])

def init_data():
    # Define data loaders for training and testing
    train_dataset = CustomImageFolder(train_root, transform=eval_transform, fixed_size=fixed_train_size)
    val_dataset = CustomImageFolder(val_root, transform=eval_transform, fixed_size=fixed_valid_size)
    test_dataset = CustomImageFolder(test_root, transform=eval_transform, fixed_size=fixed_test_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def loop(model, loader, criterion, optimizer, device, phase="training"):
    if phase == "training":
        model.train()
    else:
        model.eval()
    
    # Initialize for IoU calculation
    total_TP, total_TN, total_FP, total_FN, iou = 0, 0, 0, 0, 0
    total_loss = 0.0

    with torch.set_grad_enabled(phase == "training"):
        for images, masks in loader:
            # Model prediction and any necessary processing here
            images, masks = images.to(device), masks.to(device)

            # Handle different model output types
            model_output = model(images)
            outputs = model_output if isinstance(model_output, torch.Tensor) else model_output['out']
            
            loss = criterion(outputs, masks)

            if phase == "training":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            # Convert outputs and masks to binary if necessary, e.g., for segmentation tasks
            output_bin = (outputs > 0.5).float()
            mask_bin = masks.float()  # Assuming masks are already binary
            
            # Calculate TP, FP, FN (and TN if needed) for the current batch
            TP = ((output_bin == 1) & (mask_bin == 1)).sum().item()
            TN = ((output_bin == 0) & (mask_bin == 0)).sum().item()
            FP = ((output_bin == 1) & (mask_bin == 0)).sum().item()
            FN = ((output_bin == 0) & (mask_bin == 1)).sum().item()
            
            # Accumulate metrics components
            total_TP += TP
            total_TN += TN
            total_FP += FP
            total_FN += FN

            if phase == "testing":
                # iterate through imgs, masks and outputs to plot them
                for i in range(0, len(outputs)):
                    visualize_segmentation(images[i], masks[i], outputs[i], image_name=f"output{i}_")

        # Calculate metrics using the accumulated values
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN) if (total_TP + total_TN + total_FP + total_FN) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FP) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
    
    avg_loss = total_loss / len(loader.dataset)

    return {
        'iou': iou,
        'avg_loss': avg_loss,
        'accuracy': accuracy, 
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def main():
    # Check if GPU is available
    print("is cuda available?:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    model_name = config['Model']['name']
    train_loader, val_loader, test_loader = init_data()
    
    if model_name == 'unet':
        model = init_unet_model(device)
    elif model_name == 'unet_colab':
        model = init_unet_model_colab(device)
    elif model_name == 'efficientUNet_model':
        model = init_efficientUNet_model(device)
    elif model_name == 'deeplabv3_resnet50':
        model = init_deeplabv3_resnet50_model(device)
    elif model_name == 'imported_segm_model_unet_resnet34_imagenet':
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    else:
        model = init_unet_model(device)
    
    print_model_parameters(model)
    model.to(device)
    
    # questionable approach
    background_percentage = 99
    target_percentage = 1
    # Calculate pos_weight
    pos_weight_value = background_percentage / target_percentage
    pos_weight = torch.tensor([pos_weight_value])
    pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    metrics = {
        'train': {'avg_loss': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []},
        'val': {'avg_loss': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    }

    num_epochs = int(config['Model']['num_epochs'])

    start_time = time.time()  # Start timer for whole NN learning phase
    best_val_loss = float('inf') # For best model results tracking

    for epoch in range(num_epochs):
        # Training phase
        train_metrics = loop(model, train_loader, criterion, optimizer, device, "training")

        for key in metrics['train'].keys():
            metrics['train'][key].append(train_metrics[key])
        print(f"Training: {train_metrics}")
    
        # Validation phase
        val_metrics = loop(model, val_loader, criterion, None, device, phase="validation")

        for key in metrics['val'].keys():
            metrics['val'][key].append(val_metrics[key])
        print(f"Validation: {val_metrics}")

        end_time = time.time()
        minutes = (end_time - start_time) // 60
        seconds = (end_time - start_time) % 60
        
        val_loss = val_metrics['avg_loss']
        if val_metrics['avg_loss'] < best_val_loss:
            print(f"Validation loss improved from {best_val_loss} to {val_loss}")
            best_val_loss = val_loss

            model_config = f'{model_name}_{fixed_train_size}_{num_epochs}E_{batch_size}B'
            model_path =f'results/{model_config}/{model_config}_best_model.pth'
            create_folder_for_results(model_config)
            torch.save(model.state_dict(), model_path)
            print("------- Saved best model ---------")

        print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"time spent training the {model_name} NN {int(minutes)}:{int(seconds)}")

    print(print_model_parameters(model))

    for key in config['Model']:
        print(f'{key}: {config["Model"][key]}')

    for metric_name in metrics['train'].keys(): 
        plot_metrics(metrics, metric_name) # takes care of plotting val metrics as well
    
    # Test the model
    test_metrics = loop(model, test_loader, criterion, None, device, phase="testing")
    
    print(f'test_metrics: {test_metrics}')

if __name__ == "__main__":
    main()

