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
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 on the weighted binary dataset.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for train/val/test loaders.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on.")
    parser.add_argument(
        "--load-path",
        default="models/resnet18_frozen_weighted.pth",
        help="Path to the frozen ResNet18 weights to fine-tune.",
    )
    parser.add_argument(
        "--save-path",
        default="models/resnet18_finetuned_weighted.pth",
        help="Path to save the best fine-tuned model.",
    )
    return parser.parse_args()


def build_loaders(batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset, test_dataset = create_datasets()
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


def set_fine_tune_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


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
    state_path = Path(args.load_path)
    if not state_path.exists():
        raise FileNotFoundError(f"Frozen model file not found: {state_path}")
    model.load_state_dict(torch.load(state_path, map_location=device))

    set_fine_tune_parameters(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    optimizer = torch.optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": 1e-5},
            {"params": model.fc.parameters(), "lr": 1e-4},
        ]
    )

    best_val_loss = float("inf")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
        print(f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
