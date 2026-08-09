from __future__ import annotations

from pathlib import Path
from typing import Tuple

from dataset import CarDamageDataset
from transforms import evaluation_transform, train_transform


def create_datasets(
    splits_dir: Path = Path("data/splits"),
    images_dir: Path = Path("data/raw/car-damage-dataset/images"),
) -> Tuple[CarDamageDataset, CarDamageDataset, CarDamageDataset]:
    train_dataset = CarDamageDataset(
        csv_path=splits_dir / "train_labels.csv",
        images_dir=images_dir,
        transform=train_transform,
    )

    val_dataset = CarDamageDataset(
        csv_path=splits_dir / "val_labels.csv",
        images_dir=images_dir,
        transform=evaluation_transform,
    )

    test_dataset = CarDamageDataset(
        csv_path=splits_dir / "test_labels.csv",
        images_dir=images_dir,
        transform=evaluation_transform,
    )

    return train_dataset, val_dataset, test_dataset


def main() -> None:
    train_dataset, val_dataset, test_dataset = create_datasets()

    print("Datasets created:")
    print(f"  train: {len(train_dataset)} samples")
    print(f"  val:   {len(val_dataset)} samples")
    print(f"  test:  {len(test_dataset)} samples")


if __name__ == "__main__":
    main()
