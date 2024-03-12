from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
from torchvision.transforms.functional import to_pil_image
from utils.visualization import visualize_segmentation
import sys

models_arr = [
    fcn_resnet50,
    fcn_resnet101,
    deeplabv3_resnet50,
    lraspp_mobilenet_v3_large
    ]

weights_arr = [
    FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
    FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
    DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
    LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    ]

image_name = sys.argv[1]
object = "boat"
image_path = f'other_pics\\{image_name}.jpg'

img = read_image(image_path)

# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)

print(weights)

for model_class, weights in zip(models_arr, weights_arr):
    print(weights)
    model = model_class()

    # Step 1: Initialize model with the best available weights
    # weights = FCN_ResNet50_Weights.DEFAULT
    # model = fcn_resnet50(weights=weights)

    # weights = FCN_ResNet101_Weights.DEFAULT
    # model = fcn_resnet101(weights=weights)

    # weights = DeepLabV3_ResNet50_Weights.DEFAULT
    # model = deeplabv3_resnet50(weights=weights)

    # weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    # model = lraspp_mobilenet_v3_large(weights=weights)

    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    mask = normalized_masks[0, class_to_idx[object]]
    # to_pil_image(mask).save(f'C:\\Users\\Master\\Documents\\VU\\ML\\aero_pytorch\\results\\{image_name}.png')
    visualize_segmentation(img, to_pil_image(mask), image_name + "_" + model._get_name())

    # print(type(prediction), type(normalized_masks))