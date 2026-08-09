from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataloaders import create_datasets
from resnet_model import build_resnet_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen ResNet18 model on the test split.")
    parser.add_argument(
        "--model-path",
        default="models/resnet18_frozen_weighted.pth",
        help="Path to the saved ResNet model weights.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for evaluation.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on.",
    )
    parser.add_argument(
        "--output-file",
        default="reports/resnet18_frozen_weighted_results.txt",
        help="File to write evaluation results to.",
    )
    return parser.parse_args()


def build_test_loader(batch_size: int, num_workers: int) -> DataLoader:
    _, _, test_dataset = create_datasets()
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    return all_labels, all_predictions


def format_results(all_labels: list[int], all_predictions: list[int]) -> str:
    report = classification_report(all_labels, all_predictions, digits=4)
    matrix = confusion_matrix(all_labels, all_predictions)
    balanced = balanced_accuracy_score(all_labels, all_predictions)
    accuracy = sum(int(y_pred == y_true) for y_pred, y_true in zip(all_predictions, all_labels)) / len(all_labels)

    lines = [
        "Classification Report:",
        report,
        "Confusion Matrix:",
        str(matrix),
        f"Balanced accuracy: {balanced:.4f}",
        f"Overall accuracy: {accuracy:.4f}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = build_resnet_model(num_classes=2).to(device)
    state_path = Path(args.model_path)
    if not state_path.exists():
        raise FileNotFoundError(f"Model file not found: {state_path}")
    model.load_state_dict(torch.load(state_path, map_location=device))

    test_loader = build_test_loader(args.batch_size, args.num_workers)
    all_labels, all_predictions = evaluate(model, test_loader, device)

    output = format_results(all_labels, all_predictions)
    print(output)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
