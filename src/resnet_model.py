from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 2) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


build_resnet_model = build_resnet18


if __name__ == "__main__":
    model = build_resnet_model()
    print(model)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
