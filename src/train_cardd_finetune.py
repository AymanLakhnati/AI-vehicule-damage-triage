import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import tqdm

from cardd_dataloaders import build_dataloaders
from cardd_model import build_cardd_model


NUM_EPOCHS = 5
BATCH_SIZE = 32
NUM_WORKERS = 0
MODEL_PATH = Path('models/cardd_resnet18_finetuned.pth')

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


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Val', leave=False):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            running_loss += loss.item() * images.size(0)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).int()

            all_targets.append(targets.cpu())
            all_predictions.append(predictions.cpu())

    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_predictions = torch.cat(all_predictions, dim=0).numpy()

    val_loss = running_loss / len(loader.dataset)
    val_macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    return val_loss, val_macro_f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_loader, val_loader, _ = build_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = build_cardd_model().to(device)

    # Freeze everything first, then unfreeze layer4 and fc
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable:,}')

    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHTS.to(device))
    optimizer = torch.optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4},
    ])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Epoch {epoch}/{NUM_EPOCHS}')

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_macro_f1 = validate(model, val_loader, criterion, device)

        print(f'train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_macro_f1={val_macro_f1:.6f}')

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'Saved best model to {MODEL_PATH} (val_macro_f1={best_macro_f1:.6f})')

    print('Fine-tuning complete.')


if __name__ == '__main__':
    main()
