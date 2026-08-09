from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def read_labels(label_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    rows: List[Dict[str, str]] = []
    with label_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            raise ValueError(f"Label file has no header: {label_path}")
        for row in reader:
            rows.append(row)
    return rows, reader.fieldnames


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    )
    return image_files


def summarize_labels(rows: List[Dict[str, str]]) -> Dict[str, int]:
    label_counts: Dict[str, int] = {}
    for row in rows:
        label = row.get("label") or row.get("damage_type") or row.get("category") or "unknown"
        label_counts[label] = label_counts.get(label, 0) + 1
    return label_counts


def audit_dataset(dataset_path: Path) -> None:
    dataset_path = dataset_path.expanduser().resolve()
    labels_path = dataset_path / "labels.csv"
    images_dir = dataset_path / "images"

    print(f"Dataset path: {dataset_path}")
    print(f"Labels file: {labels_path}")
    print(f"Images folder: {images_dir}")
    print()

    rows, headers = read_labels(labels_path)
    image_files = list_images(images_dir)

    print(f"Labels file columns: {headers}")
    print(f"Total label rows: {len(rows)}")
    print(f"Total image files: {len(image_files)}")

    filename_column = None
    for candidate in ["image", "filename", "file_name", "image_id", "img"]:
        if candidate in [h.lower() for h in headers]:
            filename_column = candidate
            break
    if filename_column is None:
        filename_column = headers[0]

    label_counts = summarize_labels(rows)
    print(f"Label categories found: {len(label_counts)}")
    for label, count in sorted(label_counts.items(), key=lambda item: item[1], reverse=True)[:10]:
        print(f"  {label}: {count}")

    missing_images = []
    bad_rows = []
    image_names = {p.name for p in image_files}

    for row in rows:
        key = row.get(filename_column) or ""
        if not key:
            bad_rows.append(row)
            continue
        if key not in image_names:
            # support path-based values too
            key_name = Path(key).name
            if key_name not in image_names:
                missing_images.append(key)

    if bad_rows:
        print(f"Rows with missing filename values: {len(bad_rows)}")
    if missing_images:
        print(f"Labels referencing missing images: {len(missing_images)}")
        for missing in missing_images[:10]:
            print(f"  missing: {missing}")
    else:
        print("All labeled image filenames exist in the images folder.")

    extra_images = sorted(image_names - {Path(row.get(filename_column) or "").name for row in rows})
    if extra_images:
        print(f"Images not referenced in labels.csv: {len(extra_images)}")
        for extra in extra_images[:10]:
            print(f"  extra: {extra}")
    else:
        print("All image files are referenced in labels.csv.")

    print()
    print("Dataset audit completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a Kaggle dataset folder for image-label consistency.")
    parser.add_argument(
        "--dataset-path",
        default="data/raw/car-damage-dataset",
        help="Path to the downloaded dataset folder (default: data/raw/car-damage-dataset)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    audit_dataset(Path(args.dataset_path))
