from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from utils.visualization import visualize_segmentation
import sys
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, lraspp_mobilenet_v3_large
from torchvision.models.segmentation import FCN_ResNet50_Weights, FCN_ResNet101_Weights, DeepLabV3_ResNet50_Weights, LRASPP_MobileNet_V3_Large_Weights

# Define the array of models and corresponding weights
models_arr = [fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, lraspp_mobilenet_v3_large]
weights_arr = [FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1]

image_name = sys.argv[1]
object_name = "boat"  # Specify the object name here
image_path = f'other_pics\\{image_name}.jpg'
img = read_image(image_path)

for model_class, weights in zip(models_arr, weights_arr):
    # Initialize the model with the specified weights
    model = model_class(pretrained=True, weights=weights)
    model.eval()

    # Initialize the inference transforms
    preprocess = weights.transforms()

    # Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Use the model and visualize the prediction
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    mask = normalized_masks[0, class_to_idx[object_name]]

    # Visualize the segmentation result
    visualize_segmentation(img, to_pil_image(mask), f'{image_name}_{model._get_name()}_{object_name}')
