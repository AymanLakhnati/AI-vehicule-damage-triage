import os
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from cardd_dataloaders import build_dataloaders
from cardd_model import build_cardd_model


NUM_EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 0
MODEL_PATH = Path('models/cardd_resnet18_frozen.pth')

POS_WEIGHTS = torch.tensor([
    1.2673,
    0.8686,
    5.4885,
    5.0043,
    4.7587,
    11.8584,
], dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc='Train', leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Val', leave=False):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_loader, val_loader, test_loader = build_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = build_cardd_model().to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable:,}')

    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHTS.to(device))
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Epoch {epoch}/{NUM_EPOCHS}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'train_loss={train_loss:.6f}, val_loss={val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'Saved best model to {MODEL_PATH} (val_loss={best_val_loss:.6f})')

    print('Training complete.')

    # Also run one test batch printout as a sanity check
    images, targets = next(iter(train_loader))
    print('Images:', images.shape)
    print('Targets:', targets.shape)
    print('Example target:', targets[0])


if __name__ == '__main__':
    main()
