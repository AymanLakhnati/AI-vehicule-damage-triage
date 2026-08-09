import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from cardd_model import build_cardd_model
from cardd_dataset import CarDDMultiLabelDataset
from transforms import evaluation_transform


CLASSES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]

MODEL_PATH = Path('models/cardd_resnet18_frozen.pth')
REPORT_PATH = Path('reports/cardd_resnet18_frozen_results.txt')


def load_model(device):
    model = build_cardd_model().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_test_loader(batch_size=32, num_workers=0):
    ann_path = Path('data/raw/cardd/CarDD_release/CarDD_COCO/annotations/instances_test2017.json')
    images_dir = Path('data/raw/cardd/CarDD_release/CarDD_COCO/test2017')
    dataset = CarDDMultiLabelDataset(ann_path, images_dir, transform=evaluation_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def evaluate(model, loader, device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).int()

            y_true.append(targets.cpu().numpy())
            y_pred.append(predictions.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    return y_true, y_pred


def format_report(y_true, y_pred):
    report_lines = []
    report_lines.append('CarDD Frozen ResNet18 Evaluation')
    report_lines.append('==================================')
    report_lines.append('')

    report_lines.append('Classification report:')
    report_lines.append(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    report_lines.append('')

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    report_lines.append(f'Macro F1: {macro_f1:.6f}')
    report_lines.append(f'Micro F1: {micro_f1:.6f}')
    report_lines.append('')

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    report_lines.append('Per-class precision, recall, F1:')
    for cls, p, r, f in zip(CLASSES, precision, recall, f1):
        report_lines.append(f'{cls}: precision={p:.6f}, recall={r:.6f}, f1={f:.6f}')

    return '\n'.join(report_lines)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    loader = build_test_loader()

    y_true, y_pred = evaluate(model, loader, device)
    report_text = format_report(y_true, y_pred)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding='utf-8')
    print(report_text)
    print(f'Wrote report to {REPORT_PATH}')


if __name__ == '__main__':
    main()
