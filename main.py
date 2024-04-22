import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import configparser
import time
from utils.dataLoading import CustomImageFolder
from utils.visualization import visualize_batch, plot_metrics, create_folder_for_results, append_results_to_csv, save_results_to_csv
from models.unet import init_unet_model
from models.unet_colab import init_unet_model_colab
from models.deeplabv3_resnet50 import init_deeplabv3_resnet50_model
import segmentation_models_pytorch as smp
import argparse
from utils.earlyStopping import EarlyStopping

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

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--fixed_train_size", type=int, default=0, help="fixed_train_size")
parser.add_argument("--fixed_valid_size", type=int, default=0, help="fixed_valid_size")
parser.add_argument("--fixed_test_size", type=int, default=0, help="fixed_test_size")
parser.add_argument("--batch_size", type=int, default=0, help="batch_size")
parser.add_argument("--num_epochs", type=int, default=0, help="num_epochs")
parser.add_argument("--model_name", type=int, default=0, help="model_name")

args = parser.parse_args()

# Read train_root from env.config file
config = configparser.ConfigParser()
config.read('env.config')
train_root = config['Paths']['train_root']
val_root = config['Paths']['val_root']
test_root = config['Paths']['test_root']
early_stopping_enabled = config['Model']['early_stopping_enabled']
fixed_train_size = args.fixed_train_size if args.fixed_train_size else int(config['Model']['fixed_train_size'])
fixed_valid_size = args.fixed_valid_size if args.fixed_valid_size else int(config['Model']['fixed_valid_size'])
fixed_test_size = args.fixed_test_size if args.fixed_test_size else int(config['Model']['fixed_test_size'])
batch_size = args.batch_size if args.batch_size else int(config['Model']['batch_size'])
num_epochs = args.num_epochs if args.num_epochs else int(config['Model']['num_epochs'])
model_name =  args.model_name if args.model_name else config['Model']['name']


save_path = create_folder_for_results(f'{model_name}_{fixed_train_size}_{num_epochs}E_{batch_size}B')
print(f"save_path: {save_path}")

print(f"fixed_train_size: {fixed_train_size}")
print(f"fixed_valid_size: {fixed_valid_size}")
print(f"fixed_test_size: {fixed_test_size}")
print(f"batch_size: {batch_size}")
print(f"num_epochs: {num_epochs}")

def init_data():
    torch.manual_seed(0) # to reproduce the same results (avoid random img selection)
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
                visualize_batch(images, masks, outputs, save_path)
                # iterate through imgs, masks and outputs to plot them
                # for i in range(0, len(outputs)):
                #     visualize_segmentation(images[i], masks[i], outputs[i], image_name=f"output{i}_")

        # Calculate metrics using the accumulated values
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN) if (total_TP + total_TN + total_FP + total_FN) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FP) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
        # dice = 2 * total_TP / (2 * total_TP + total_FP + total_FN) if (2 * total_TP + total_FP + total_FN) > 0 else 0
    
    avg_loss = total_loss / len(loader.dataset)

    metrics = {
        'iou': iou,
        # 'dice': dice,
        'avg_loss': avg_loss,
        'accuracy': accuracy, 
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return {key: round(value, 4) for key, value in metrics.items()}

def main():
    early_stopping = EarlyStopping(patience=10, verbose=True)  # Setting patience to 10 for example

    # Check if GPU is available
    print("is cuda available?:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = init_data()
    
    if model_name == 'unet_baseline':
        model = init_unet_model(device)
    elif model_name == 'unet_colab':
        model = init_unet_model_colab(device)
    elif model_name == 'deeplabv3_resnet50':
        model = init_deeplabv3_resnet50_model(device)
    elif model_name == 'unet_resnet34_imagenet':
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    elif model_name == 'unet_efficientnet_imagenet':
        model = smp.Unet(encoder_name="efficientnet-b1", encoder_weights="imagenet", in_channels=3, classes=1)
    else:
        model = init_unet_model(device)
    
    model.to(device)
    
    # questionable approach
    background_percentage = 90
    target_percentage = 10
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

    start_time = time.time()  # Start timer for whole NN learning phase
    best_val_loss = float('inf') # For best model results tracking
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # Training phase
        train_metrics = loop(model, train_loader, criterion, optimizer, device, phase="training")

        for key in metrics['train'].keys():
            metrics['train'][key].append(train_metrics[key])
        print(f"Training: {train_metrics}")
    
        # Validation phase
        val_metrics = loop(model, val_loader, criterion, None, device, phase="validation")

        for key in metrics['val'].keys():
            metrics['val'][key].append(val_metrics[key])
        print(f"Validation: {val_metrics}")

        end_time = time.time()
        minutes = int((end_time - start_time) // 60)
        seconds = int((end_time - start_time) % 60)
        formatted_time = str(minutes) + ":" + str(seconds).zfill(2)
        
        val_loss = val_metrics['avg_loss']
        if val_metrics['avg_loss'] < best_val_loss:
            print(f"Validation loss improved from {best_val_loss} to {val_loss}")

            best_val_loss = val_loss # should save best val_loss to csv


            # # currently not in use for experiments
            # model_config = f'{model_name}_{fixed_train_size}_{num_epochs}E_{batch_size}B'
            # model_path =f'results/{model_config}/{model_config}_best_model.pth'
            # create_folder_for_results(model_config)
            # torch.save(model.state_dict(), model_path)
            # print("------- Saved best model ---------")

        # if early_stopping_enabled:
        #     early_stopping(val_metrics['avg_loss'])
        #     best_epoch = epoch

        #     model_config = f'{model_name}_{fixed_train_size}_{best_epoch}E_{batch_size}B'
        #     model_save_path = create_folder_for_results(f'{model_config}')
        #     model_save_path = f'{model_save_path}/{model_config}_best_model.pth'
        #     torch.save(model.state_dict(), model_save_path)

        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break
        
    print(f"time spent training the {model_name} NN {int(minutes)}:{int(seconds)}")

    for key in config['Model']:
        print(f'{key}: {config["Model"][key]}')

    for metric_name in metrics['train'].keys(): 
        plot_metrics(metrics, metric_name, save_path, caption=f'{model_name}_{fixed_train_size}_{num_epochs}E_{batch_size}B') # takes care of plotting val metrics as well
    
    save_results_to_csv(metrics['train'], save_path, 'train')
    save_results_to_csv(metrics['val'], save_path, 'val')
    
    # Test the model
    test_metrics = loop(model, test_loader, criterion, None, device, phase="testing")

    model_test_results = {
        "model_name": model_name,
        "time": formatted_time,
        "epochs": num_epochs,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "fixed_train_size": fixed_train_size,
        "fixed_valid_size": fixed_valid_size,
        "fixed_test_size": fixed_test_size,
        "batch_size": batch_size,
        "training_time": formatted_time,
        **test_metrics}

    print(model_test_results)

    # for result in model_test_results:
    #     print(f'{result}: {model_test_results[result]}')
    
    save_results_to_csv(model_test_results, save_path, 'model_test_results')
    append_results_to_csv(model_test_results)

if __name__ == "__main__":
    main()
