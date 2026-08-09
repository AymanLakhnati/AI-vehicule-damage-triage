import json
from pathlib import Path

import torch
from sklearn.metrics import f1_score

from cardd_dataloaders import build_dataloaders
from cardd_model import build_cardd_model

CLASSES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]

MODEL_PATH = Path('models/cardd_resnet18_finetuned.pth')
OUTPUT_PATH = Path('models/cardd_thresholds.json')

THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90,
]


def load_validation_predictions(model, loader, device):
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.sigmoid(logits).cpu()
            all_probs.append(probabilities)
            all_targets.append(targets)

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_probs = torch.cat(all_probs, dim=0).numpy()
    return y_true, y_probs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    _, val_loader, _ = build_dataloaders(batch_size=32, num_workers=0)
    model = build_cardd_model().to(device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model weights not found: {MODEL_PATH}')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    y_true, y_probs = load_validation_predictions(model, val_loader, device)

    thresholds = {}
    for class_idx, class_name in enumerate(CLASSES):
        best_threshold = None
        best_f1 = -1.0
        for threshold in THRESHOLDS:
            y_pred = (y_probs[:, class_idx] >= threshold).astype(int)
            f1 = f1_score(y_true[:, class_idx], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[class_name] = best_threshold
        print(f"{class_name}: best_threshold={best_threshold:.2f}, F1={best_f1:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open('w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved thresholds to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
