import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, \
    alexnet, shufflenet_v2_x0_5
import torchvision.transforms as T
from PIL import Image
import warnings
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

warnings.filterwarnings("ignore")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_label_map():
    label_map = {
        "call": 1, "like": 2, "peace": 3, "stop": 4, "no_gesture": 5
    }

    return label_map


def get_label_map_reverse():
    reverse_label_map = {}

    label_map = get_label_map()

    for key in label_map:
        value = label_map[key]
        reverse_label_map[value] = key

    return reverse_label_map


def compute_iou_per_box(boxA, boxB):
    # Compute the intersection over union of two boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def get_faster_rcnn_model(num_classes=5, backbone_type='resnet18', pretrained=True):
    """
    Returns a Faster-RCNN model with a specified backbone (ResNet, MobileNetV2/V3, AlexNet).

    Args:
        num_classes (int): Number of classes for the detection task.
        backbone_type (str): Backbone type ('resnet18', 'resnet34', 'resnet50', 'mobilenet_v2',
                             'mobilenet_v3_large', 'mobilenet_v3_small', 'alexnet').
        pretrained (bool): If True, use pretrained weights for the backbone.

    Returns:
        model (nn.Module): Faster-RCNN model with the specified backbone.
    """

    # Load the backbone based on the backbone_type argument
    if backbone_type == 'resnet18':
        backbone = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        backbone_out_channels = 512

    elif backbone_type == 'resnet34':
        backbone = resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        backbone_out_channels = 512

    elif backbone_type == 'resnet50':
        backbone = resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        backbone_out_channels = 2048

    elif backbone_type == 'mobilenet_v2':
        backbone = mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None).features
        backbone_out_channels = 1280  # Feature channels from MobileNetV2

    elif backbone_type == 'mobilenet_v3_large':
        backbone = mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None).features
        backbone_out_channels = 960  # Feature channels from MobileNetV3 Large

    elif backbone_type == 'mobilenet_v3_small':
        backbone = mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None).features
        backbone_out_channels = 576  # Feature channels from MobileNetV3 Small

    elif backbone_type == 'alexnet':
        backbone = alexnet(weights='IMAGENET1K_V1' if pretrained else None).features
        backbone_out_channels = 256  # Feature channels from AlexNet

    elif backbone_type == 'shufflenet_v2_x0_5':
        backbone = shufflenet_v2_x0_5(weights='IMAGENET1K_V1' if pretrained else None)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        backbone_out_channels = 1024

    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")

    # For MobileNet and AlexNet, we keep the backbone as it is since it's already suitable for detection tasks.
    if backbone_type.startswith('resnet'):
        # Remove the fully connected layers for ResNet
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    # Set the output channels for the backbone
    backbone.out_channels = backbone_out_channels

    # Define the RPN anchor generator (Region Proposal Network)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * len(((32, 64, 128, 256, 512),))
    )

    # Define the RoI pooling layer (Region of Interest)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    # Create the Faster-RCNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model


# Function to load model from saved state
def load_model(backbone_name, model_path, device):
    model = get_faster_rcnn_model(num_classes=5, backbone_type=backbone_name)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()  # Set model to evaluation mode
    return model


def visualize_predictions(image, predictions, targets=None):
    """
    Visualize ground-truth and predicted bounding boxes on a few sample images.

    Args:
        image (Tensor): Input images (batch of images).
        predictions (dict): List of model predictions (boxes, labels).
        targets (dict): List of ground truth annotations (boxes, labels).
    """
    label_map = get_label_map()
    reverse_label_map = get_label_map_reverse()
    confidence_score_threshold = 0.3
    iou_threshold = 0.5

    plt.figure(figsize=(12, 8))

    image = image.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) for plotting
    pred_boxes = predictions["boxes"].detach().numpy()
    pred_scores = predictions["scores"].detach().numpy()
    pred_labels = predictions["labels"].detach().numpy()

    gt_boxes = []
    gt_labels = []
    if targets:
        gt_boxes = targets["boxes"].detach().numpy()
        gt_labels = targets["labels"].detach().numpy()

    plt.imshow(image)

    # Plot predicted boxes
    height = 0
    drawn_predictions = []
    for box, score, pred_label_int in zip(pred_boxes, pred_scores, pred_labels):
        box = box.tolist()
        pred_label_int = pred_label_int.item()
        over_iou_thresh = False
        for other_box, other_box_label in drawn_predictions:
            iou = compute_iou_per_box(box, other_box)
            if iou > iou_threshold and other_box_label == pred_label_int:
                over_iou_thresh = True
                break

        drawn_predictions.append((box, pred_label_int))

        if not over_iou_thresh:
            box_string = f'Predicted ({reverse_label_map[pred_label_int]} - {score:.2f})'

            colour = "red" if score > confidence_score_threshold else "yellow"
            linewidth = 2 if score > confidence_score_threshold else 1

            plt.gca().add_patch(
                plt.Rectangle(
                    xy=(box[0], box[1])
                    , width=(box[2] - box[0])
                    , height=(box[3] - box[1])
                    , fill=False
                    , edgecolor=colour
                    , linewidth=linewidth
                    , label=box_string
                )
            )

            plt.text(
                x=box[0]
                , y=box[1] + height
                , s=box_string
                , fontdict=dict(color=colour)
            )

            if height == 0:
                height += (box[3] - box[1])
            else:
                height += 20

    # Plot ground-truth boxes
    height = 0
    for gt_box, gt_label_int in zip(gt_boxes, gt_labels):
        gt_label = gt_label_int.item()
        gt_box = gt_box.tolist()
        gt_box_string = f'Ground Truth ({reverse_label_map[gt_label_int]})'
        plt.gca().add_patch(
            plt.Rectangle(
                xy=(gt_box[0], gt_box[1])
                , width=(gt_box[2] - gt_box[0])
                , height=(gt_box[3] - gt_box[1])
                , fill=False
                , edgecolor='green'
                , linewidth=2
                , label=gt_box_string))

        plt.text(
            x=gt_box[0] + (gt_box[2] - gt_box[0])
            , y=gt_box[1] + height + 20
            , s=gt_box_string
            , fontdict=dict(color='green')
        )

    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Training
num_classes = 6
root_dir = './data/hagrid-sample-120k-384p/hagrid_120k'
backbone_type = 'mobilenet_v3_small'

# Mean and std used in normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

model_paths = {
    'mobilenet_v3_small': './saved_models/faster_rcnn_mobilenet_v3_small.pth',
    # 'mobilenet_v3_small': './saved_models/faster_rcnn_mobilenet_v3_small_T_23-10-24_02-25.pth', # OLD
    # 'mobilenet_v3_large': './saved_models/faster_rcnn_mobilenet_v3_large_T_23-10-24_11-33.pth',
    # 'mobilenet_v2': './saved_models/faster_rcnn_mobilenet_v2_T_23-10-24_20-25.pth',
    'alexnet': './saved_models/faster_rcnn_alexnet_T_23-10-24_13-44.pth',
    'resnet18': './saved_models/faster_rcnn_resnet18_T_16-10-24_23-26.pth',
    'resnet34': './saved_models/faster_rcnn_resnet34_T_17-10-24_05-54.pth',
    'resnet50': './saved_models/faster_rcnn_resnet50_T_20-10-24_15-15_VL_0.02290_EP_47.pth',
    # 'shufflenet_v2_x0_5': './saved_models/faster_rcnn_shufflenet_v2_x0_5_T_24-10-24_17-08_VL_0.04067_EP_20.pth',

}

image_path = './data/hagrid-sample-120k-384p/hagrid_120k/train_val_ok/0a1ffcff-bbfc-45a1-827c-69b69eba1591.jpg'

model = load_model(backbone_name=backbone_type, model_path=model_paths[backbone_type], device=device)

image = Image.open(image_path).convert("RGB")

# Print debug info about the image
if isinstance(image, np.ndarray):
    print(f"Image dtype: {image.dtype}, shape: {image.shape}")

# Convert to PIL Image if needed
if isinstance(image, np.ndarray) and image.ndim == 3:
    image = Image.fromarray(image)

# Ensure the image is in RGB mode
if image.mode != 'RGB':
    print(f"Converting image mode from {image.mode} to RGB")
    image = image.convert('RGB')

transforms = T.Compose([
    T.Resize((512, 384)),
    # T.Lambda(lambda img: resize_if_needed(img, target_size=(512, 384))),  # Conditionally resize
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

if transforms:
    image = transforms(image)

# inference_times = []
# for i in tqdm(range(50), desc="Performing Inference"):
#     # Forward pass
#     time_start = time.time()
#     predictions = model(image.unsqueeze(0))
#     time_end = time.time()
#     inference_times.append(time_end - time_start)

predictions = model(image.unsqueeze(0))
visualize_predictions(image, predictions[0])

# plt.scatter(range(len(inference_times)), inference_times)
# plt.title(f"Inference Times: ({np.mean(inference_times):.2f}s)")
# plt.show()