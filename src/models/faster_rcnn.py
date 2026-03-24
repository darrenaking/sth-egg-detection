from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(config):
    # fasterrcnn_resnet50_fpn includes an internal GeneralizedRCNNTransform that
    # resizes images (min_size=800, max_size=1333) and applies ImageNet normalization.
    # Augmentations operate on original-resolution images; do not add Normalize.
    model = fasterrcnn_resnet50_fpn(
        weights=config["model"]["pretrained_weights"],
        trainable_backbone_layers=config["model"]["trainable_backbone_layers"],
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, config["model"]["num_classes"]
    )

    return model