import json
import torch
from pathlib import Path
from sklearn.metrics import classification_report, f1_score

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
THRESHOLDS_PATH = Path('models/cardd_thresholds.json')
OUTPUT_PATH = Path('reports/cardd_resnet18_thresholded_results.txt')


def evaluate(model, loader, threshold_tensor, device):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold_tensor).int().cpu()
            all_targets.append(targets)
            all_predictions.append(predictions)

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_pred = torch.cat(all_predictions, dim=0).numpy()
    return y_true, y_pred


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    _, _, test_loader = build_dataloaders(batch_size=32, num_workers=0)
    model = build_cardd_model().to(device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model weights not found: {MODEL_PATH}')
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f'Threshold file not found: {THRESHOLDS_PATH}')

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    with THRESHOLDS_PATH.open('r', encoding='utf-8') as f:
        thresholds = json.load(f)

    threshold_tensor = torch.tensor(
        [
            thresholds['dent'],
            thresholds['scratch'],
            thresholds['crack'],
            thresholds['glass shatter'],
            thresholds['lamp broken'],
            thresholds['tire flat'],
        ],
        device=device,
    )

    y_true, y_pred = evaluate(model, test_loader, threshold_tensor, device)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        zero_division=0,
        output_dict=True,
    )

    lines = []
    for cls in CLASSES:
        cls_report = report[cls]
        lines.append(
            f"{cls}: precision={cls_report['precision']:.4f}, recall={cls_report['recall']:.4f}, f1={cls_report['f1-score']:.4f}"
        )
    lines.append('')
    lines.append(f"Macro F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    lines.append(f"Micro F1: {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text('\n'.join(lines), encoding='utf-8')

    print('\n'.join(lines))
    print(f'Wrote results to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
