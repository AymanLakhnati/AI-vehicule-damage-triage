from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import matplotlib.pyplot as plt


def load_labels(csv_path: Path, label_column: str) -> Tuple[List[Dict[str, str]], List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Label file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            raise ValueError(f"Label file has no header: {csv_path}")

        fieldnames = reader.fieldnames
        header_map = {name.lower(): name for name in fieldnames}
        if label_column.lower() not in header_map:
            raise ValueError(
                f"Label column '{label_column}' not found in {fieldnames}"
            )
        rows = [dict(row) for row in reader]

    return rows, fieldnames


def get_examples(
    rows: List[Dict[str, str]],
    images_dir: Path,
    label_column: str,
    samples_per_class: int,
    seed: int,
) -> List[Tuple[Path, str, str]]:
    label_column_name = next(
        key for key in rows[0].keys() if key.lower() == label_column.lower()
    )
    image_column_name = None
    for candidate in ["image", "filename", "file_name", "image_id", "img"]:
        candidate_key = next(
            (key for key in rows[0].keys() if key.lower() == candidate),
            None,
        )
        if candidate_key is not None:
            image_column_name = candidate_key
            break
    if image_column_name is None:
        raise ValueError(
            "No image filename column found. Expected one of: image, filename, file_name, image_id, img"
        )

    grouped: Dict[str, List[Tuple[Path, str, str]]] = {}
    for row in rows:
        label = row[label_column_name]
        image_value = row.get(image_column_name, "")
        if not image_value:
            continue
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = images_dir / image_path.name
        grouped.setdefault(label, []).append((image_path, label, image_path.name))

    random.seed(seed)
    examples: List[Tuple[Path, str, str]] = []
    for label in sorted(grouped.keys(), key=str):
        subset = grouped[label]
        random.shuffle(subset)
        examples.extend(subset[:samples_per_class])

    return examples


def plot_examples(examples: List[Tuple[Path, str, str]]) -> None:
    count = len(examples)
    if count == 0:
        raise ValueError("No examples available to plot.")

    ncols = 4
    nrows = (count + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))
    axes_list = [ax for row in axes for ax in row] if nrows > 1 else [axes]  # type: ignore

    for ax, (image_path, label, filename) in zip(axes_list, examples):
        if not image_path.exists():
            ax.text(0.5, 0.5, f"Missing:\n{filename}", ha="center", va="center")
            ax.axis("off")
            continue

        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"label={label}\n{filename}", fontsize=10)
        ax.axis("off")

    for ax in axes_list[len(examples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize labeled examples from the car damage dataset."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/raw/car-damage-dataset",
        help="Path to the downloaded dataset folder containing labels.csv and images/",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the label column to use for selections.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=8,
        help="Number of examples to show for each label.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    labels_path = dataset_path / "labels.csv"
    images_dir = dataset_path / "images"

    rows, _ = load_labels(labels_path, args.label_column)
    examples = get_examples(
        rows,
        images_dir,
        args.label_column,
        args.samples_per_class,
        args.random_state,
    )

    plot_examples(examples)
