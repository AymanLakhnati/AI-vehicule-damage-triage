from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloaders import create_datasets
from resnet_model import build_resnet_model


CLASS_WEIGHTS = torch.tensor([0.5304877758026123, 8.699999809265137], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen ResNet18 with class-weighted loss.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for train/val/test loaders.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on.")
    parser.add_argument("--save-path", default="models/resnet18_frozen_weighted.pth", help="Path to save the best model.")
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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader, _ = build_loaders(args.batch_size, args.num_workers)

    model = build_resnet_model(num_classes=2).to(device)
    class_weights = CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    best_val_loss = float("inf")
    best_path = Path(args.save_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
        print(f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
