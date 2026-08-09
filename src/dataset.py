from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class CarDamageDataset(Dataset):
    def __init__(
        self,
        csv_path: Path | str,
        images_dir: Path | str,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        label_column: str = "label",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_dir}")

        self.samples: List[Tuple[Path, Any]] = []
        self.fieldnames: List[str] = []

        with self.csv_path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header: {self.csv_path}")

            self.fieldnames = reader.fieldnames
            header_map = {name.lower(): name for name in self.fieldnames}

            if label_column.lower() not in header_map:
                raise ValueError(
                    f"Label column '{label_column}' not found in {self.fieldnames}"
                )

            label_name = header_map[label_column.lower()]
            image_name = None
            for candidate in ["image", "filename", "file_name", "image_id", "img"]:
                if candidate in header_map:
                    image_name = header_map[candidate]
                    break
            if image_name is None:
                raise ValueError(
                    f"No image filename column found in {self.fieldnames}."
                )

            for row in reader:
                image_value = row.get(image_name, "")
                if not image_value:
                    continue

                image_path = Path(image_value)
                if not image_path.is_absolute():
                    image_path = self.images_dir / image_path.name

                self.samples.append((image_path, row[label_name]))

        if not self.samples:
            raise ValueError(f"No samples loaded from {self.csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_path, label = self.samples[index]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        try:
            label_value = int(label)
        except ValueError:
            label_value = label

        return image, label_value
