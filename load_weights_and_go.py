import torch
import models.unet as unet
from visualization import visualize_segmentation

def loop(model, loader, criterion, optimizer, device, phase="training"):
    if phase == "training":
        model.train()
    else:
        model.eval()
    
    # Initialize for IoU calculation
    total_TP, total_FP, total_FN, iou = 0, 0, 0, 0
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
            FP = ((output_bin == 1) & (mask_bin == 0)).sum().item()
            FN = ((output_bin == 0) & (mask_bin == 1)).sum().item()
            
            # Accumulate metrics components
            total_TP += TP
            total_FP += FP
            total_FN += FN

            if phase == "testing":
                print(phase)
                # iterate through imgs, masks and outputs to plot them
                for i in range(0, len(outputs)):
                    visualize_segmentation(images[i], masks[i], outputs[i], image_name=f"output{i}_")
        
        # Calculate metrics using the accumulated values
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
    
    avg_loss = total_loss / len(loader.dataset)

    return {
        'iou': iou,
        'avg_loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


# Initialize your model instance.
num_class = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = unet.UNet(num_class).to(device)

# Path to your saved weights file.
weights_path = 'saved_model_weights/best_model_weights_unet_1000.pth'

# Load the state dictionary from the file.
if torch.cuda.is_available():
    state_dict = torch.load(weights_path)
else:
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

# Apply the state dictionary to your model.
model.load_state_dict(state_dict)

print("Model weights loaded successfully.")


