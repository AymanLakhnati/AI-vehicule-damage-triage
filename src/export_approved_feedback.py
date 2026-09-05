import argparse
import csv
import json
import shutil
from pathlib import Path

try:
    from storage import Storage
except ModuleNotFoundError:
    from src.storage import Storage

CLASS_NAMES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]


def parse_labels(value):
    labels = {label.strip().lower() for label in value.split(",") if label.strip()}
    if labels == {"none"}:
        return set()
    unknown = labels.difference(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unknown labels: {', '.join(sorted(unknown))}")
    return labels


def export_feedback(feedback_dir, output_dir, database_path=None):
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []

    if database_path and database_path.exists():
        records = Storage(database_path).list_feedback("approved")
    else:
        records = [
            json.loads(metadata_path.read_text(encoding="utf-8"))
            for metadata_path in sorted(feedback_dir.glob("*.json"))
            if json.loads(metadata_path.read_text(encoding="utf-8")).get("status") == "approved"
        ]

    for metadata in records:
        if not metadata.get("approved_labels"):
            skipped.append(metadata.get("id", "unknown"))
            continue

        try:
            labels = parse_labels(metadata["approved_labels"])
        except ValueError:
            skipped.append(metadata.get("id", "unknown"))
            continue

        source_image = feedback_dir / f"{metadata['id']}.jpg"
        if not source_image.exists():
            skipped.append(metadata.get("id", "unknown"))
            continue

        destination_name = f"{metadata['id']}.jpg"
        shutil.copy2(source_image, output_images / destination_name)
        row = {"file_name": destination_name}
        row.update({class_name: int(class_name in labels) for class_name in CLASS_NAMES})
        rows.append(row)

    manifest = output_dir / "labels.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", *CLASS_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    return manifest, len(rows), skipped


def main():
    parser = argparse.ArgumentParser(description="Export approved feedback into a curated multilabel dataset.")
    parser.add_argument("--feedback-dir", type=Path, default=Path("feedback_data"))
    parser.add_argument("--database", type=Path, default=Path("data/autotriage.db"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/approved_feedback"))
    args = parser.parse_args()
    if not args.feedback_dir.exists():
        raise FileNotFoundError(f"Feedback directory not found: {args.feedback_dir}")
    database = args.database if args.database.exists() else None
    manifest, count, skipped = export_feedback(args.feedback_dir, args.output_dir, database)
    print(f"Exported {count} approved examples.")
    print(f"Manifest: {manifest}")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete or invalid records.")


if __name__ == "__main__":
    main()
