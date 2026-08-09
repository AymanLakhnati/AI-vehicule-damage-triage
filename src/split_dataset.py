import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split


def read_labels(label_path: Path, label_column: str) -> Tuple[List[Dict[str, str]], List[str]]:
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with label_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            raise ValueError(f"Label file has no header: {label_path}")
        fieldnames = reader.fieldnames
        if label_column not in [h.lower() for h in fieldnames]:
            raise ValueError(f"Label column '{label_column}' not found in {fieldnames}")

        rows = [dict(row) for row in reader]
    return rows, fieldnames


def write_split(rows: List[Dict[str, str]], fieldnames: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_split(name: str, rows: List[Dict[str, str]], label_column: str) -> None:
    counts = {}
    for row in rows:
        label = row.get(label_column, "")
        counts[label] = counts.get(label, 0) + 1

    print(f"{name}: {len(rows)} rows")
    for label, count in sorted(counts.items(), key=lambda item: item[0]):
        print(f"  {label}: {count}")
    print()


def split_dataset(dataset_path: Path, output_dir: Path, label_column: str, random_state: int) -> None:
    dataset_path = dataset_path.resolve()
    labels_path = dataset_path / "labels.csv"
    rows, fieldnames = read_labels(labels_path, label_column)
    label_column_name = next(h for h in fieldnames if h.lower() == label_column)

    labels = [row[label_column_name] for row in rows]

    train_rows, temp_rows, train_labels, temp_labels = train_test_split(
        rows,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=random_state,
    )

    val_rows, test_rows, val_labels, test_labels = train_test_split(
        temp_rows,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=random_state,
    )

    output_dir = output_dir.resolve()
    write_split(train_rows, fieldnames, output_dir / "train_labels.csv")
    write_split(val_rows, fieldnames, output_dir / "val_labels.csv")
    write_split(test_rows, fieldnames, output_dir / "test_labels.csv")

    print(f"Saved train/validation/test label splits to {output_dir}")
    print()
    summarize_split("Train", train_rows, label_column_name)
    summarize_split("Validation", val_rows, label_column_name)
    summarize_split("Test", test_rows, label_column_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split labels.csv into stratified train/val/test sets.")
    parser.add_argument(
        "--dataset-path",
        default="data/raw/car-damage-dataset",
        help="Path to the downloaded dataset folder containing labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits",
        help="Directory to write the split CSV files",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the label column to use for stratification",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(Path(args.dataset_path), Path(args.output_dir), args.label_column.lower(), args.random_state)

