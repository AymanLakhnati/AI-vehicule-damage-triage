import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 7  # background + 6 CarDD classes


def build_detector():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES,
    )

    return model


if __name__ == "__main__":
    model = build_detector()

    print(model.roi_heads.box_predictor)

    model.eval()

    dummy_image = torch.rand(3, 500, 700)

    with torch.no_grad():
        predictions = model([dummy_image])

    print("Prediction keys:", predictions[0].keys())
    print("Boxes:", predictions[0]["boxes"].shape)
    print("Labels:", predictions[0]["labels"].shape)
    print("Scores:", predictions[0]["scores"].shape)
