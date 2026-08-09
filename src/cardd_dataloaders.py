import torch
from torch.utils.data import DataLoader
from pathlib import Path

from transforms import train_transform, evaluation_transform
from cardd_dataset import CarDDMultiLabelDataset


def build_dataloaders(batch_size=32, num_workers=0):
    base = Path('data/raw/cardd/CarDD_release/CarDD_COCO')
    ann_dir = base / 'annotations'

    train_ds = CarDDMultiLabelDataset(ann_dir / 'instances_train2017.json', base / 'train2017', transform=train_transform)
    val_ds = CarDDMultiLabelDataset(ann_dir / 'instances_val2017.json', base / 'val2017', transform=evaluation_transform)
    test_ds = CarDDMultiLabelDataset(ann_dir / 'instances_test2017.json', base / 'test2017', transform=evaluation_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = build_dataloaders(batch_size=32, num_workers=0)
    images, targets = next(iter(train_loader))
    print('Images:', images.shape)
    print('Targets:', targets.shape)
    print('Example target:', targets[0])
