from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline_model import build_baseline_model
from dataloaders import create_datasets


def compute_class_weights(dataset: torch.utils.data.Dataset, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, label in dataset:
        counts[int(label)] += 1.0
    weights = counts.sum() / (num_classes * counts.clamp(min=1.0))
    return weights


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

    return total_loss / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline CNN on car damage data.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    parser.add_argument("--save-path", default="models/baseline.pth", help="Path to save the final model state.")
    parser.add_argument("--class-weighted", action="store_true", help="Use class-weighted CrossEntropyLoss.")
    return parser.parse_args()


def build_loaders(batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader, test_loader = build_loaders(args.batch_size, args.num_workers)
    model = build_baseline_model(num_classes=2).to(device)

    if args.class_weighted:
        class_weights = compute_class_weights(train_loader.dataset, num_classes=2).to(device)
        print(f"Using class weights: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("Starting training")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()
