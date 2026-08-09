import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 6


def build_cardd_model():
    # Use pretrained weights when available
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        # Fallback for older torchvision versions
        model = models.resnet18(pretrained=True)

    # Freeze pretrained backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace ImageNet classifier
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


if __name__ == "__main__":
    model = build_cardd_model()

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print(model.fc)
    print(f"Output shape: {output.shape}")

    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print(f"Trainable parameters: {trainable:,}")
