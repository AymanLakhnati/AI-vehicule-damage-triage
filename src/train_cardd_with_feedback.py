import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cardd_dataloaders import build_dataloaders
from cardd_model import build_cardd_model
from transforms import train_transform

CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
POS_WEIGHTS = torch.tensor([1.2673, 0.8686, 5.4885, 5.0043, 4.7587, 11.8584], dtype=torch.float32)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK = ROOT / "data" / "approved_feedback"
DEFAULT_BASE_MODEL = ROOT / "models" / "cardd_resnet18_finetuned.pth"
DEFAULT_OUTPUT = ROOT / "models" / "cardd_resnet18_feedback_finetuned.pth"
DEFAULT_RUN_REPORT = ROOT / "reports" / "feedback_training_run.json"


class ApprovedFeedbackDataset(Dataset):
    def __init__(self, manifest_path, images_dir):
        self.images_dir = Path(images_dir)
        self.rows = []
        with Path(manifest_path).open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                labels = torch.tensor([float(row[class_name]) for class_name in CLASSES], dtype=torch.float32)
                self.rows.append((row["file_name"], labels))
        if not self.rows:
            raise ValueError(f"No approved feedback rows found in {manifest_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        file_name, labels = self.rows[index]
        image_path = self.images_dir / file_name
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return train_transform(image), labels


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the classifier with approved human feedback.")
    parser.add_argument("--feedback-dir", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feedback-weight", type=float, default=3.0)
    parser.add_argument("--run-report", type=Path, default=DEFAULT_RUN_REPORT)
    return parser.parse_args()


def train_batches(model, loader, criterion, optimizer, device, loss_weight):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        (loss * loss_weight).backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def validation_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            total_loss += criterion(model(images), targets).item() * images.size(0)
    return total_loss / len(loader.dataset)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    manifest = args.feedback_dir / "labels.csv"
    feedback_images = args.feedback_dir / "images"
    for path in (manifest, feedback_images, args.base_model):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Base model: {args.base_model}")

    train_loader, val_loader, _ = build_dataloaders(batch_size=args.batch_size, num_workers=0)
    feedback_dataset = ApprovedFeedbackDataset(manifest, feedback_images)
    feedback_loader = DataLoader(feedback_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Original training images: {len(train_loader.dataset)}")
    print(f"Approved feedback images: {len(feedback_dataset)}")

    model = build_cardd_model().to(device)
    model.load_state_dict(torch.load(args.base_model, map_location=device))
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHTS.to(device))
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=2e-4)
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        original_loss = train_batches(model, train_loader, criterion, optimizer, device, 1.0)
        feedback_loss = train_batches(model, feedback_loader, criterion, optimizer, device, args.feedback_weight)
        train_loss = (original_loss * len(train_loader.dataset) + feedback_loss * len(feedback_dataset)) / (len(train_loader.dataset) + len(feedback_dataset))
        val_loss = validation_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"Saved best model to {args.output}")

    run_record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": str(args.base_model),
        "base_model_sha256": sha256_file(args.base_model),
        "output_model": str(args.output),
        "feedback_manifest": str(manifest),
        "approved_feedback_count": len(feedback_dataset),
        "original_training_count": len(train_loader.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "feedback_weight": args.feedback_weight,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "test_split_used": False,
    }
    args.run_report.parent.mkdir(parents=True, exist_ok=True)
    args.run_report.write_text(json.dumps(run_record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved training provenance to {args.run_report}")
    print("Training complete. The test split was not used.")


if __name__ == "__main__":
    main()
