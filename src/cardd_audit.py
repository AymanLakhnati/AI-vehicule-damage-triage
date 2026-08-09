import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CATEGORY_ORDER = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_split(split_path: Path):
    data = load_json(split_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_count = len(images)
    annotation_count = len(annotations)

    # category_id -> category_name
    category_map = {cat["id"]: cat["name"] for cat in categories}

    category_counter = Counter()
    image_categories = defaultdict(set)
    image_annotation_counts = Counter()

    for ann in annotations:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        category_name = category_map.get(category_id, str(category_id))
        category_counter[category_name] += 1
        image_categories[image_id].add(category_name)
        image_annotation_counts[image_id] += 1

    one_category = 0
    two_categories = 0
    three_plus_categories = 0
    multiple_instances = 0

    for image_id, categories_set in image_categories.items():
        category_count = len(categories_set)
        if category_count == 1:
            one_category += 1
        elif category_count == 2:
            two_categories += 1
        else:
            three_plus_categories += 1

        if image_annotation_counts[image_id] > 1:
            multiple_instances += 1

    # Images with zero annotations still exist in COCO; count them as zero-category images if needed.
    images_with_annotations = set(image_categories.keys())
    zero_annotation_images = image_count - len(images_with_annotations)
    one_category += zero_annotation_images

    results = {
        "image_count": image_count,
        "annotation_count": annotation_count,
        "category_counts": {name: category_counter.get(name, 0) for name in CATEGORY_ORDER},
        "one_category_images": one_category,
        "two_category_images": two_categories,
        "three_plus_category_images": three_plus_categories,
        "multiple_instances_images": multiple_instances,
        "zero_annotation_images": zero_annotation_images,
    }
    return results


def print_results(name: str, results: dict):
    print(f"=== {name} ===")
    print(f"Images: {results['image_count']}")
    print(f"Annotations: {results['annotation_count']}")
    print("Instances per class:")
    for category_name, count in results["category_counts"].items():
        print(f"  {category_name}: {count}")
    print("Images by damage category count:")
    print(f"  exactly 1 damage category: {results['one_category_images']}")
    print(f"  2 damage categories: {results['two_category_images']}")
    print(f"  3+ damage categories: {results['three_plus_category_images']}")
    print(f"Images with multiple damage instances: {results['multiple_instances_images']}")
    if results['zero_annotation_images'] > 0:
        print(f"Images with zero annotations: {results['zero_annotation_images']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Audit CarDD COCO splits for multi-label damage structure")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/raw/cardd/CarDD_release/CarDD_COCO/annotations"),
        help="Directory containing CarDD COCO annotation JSON files",
    )
    args = parser.parse_args()

    split_files = [
        "instances_train2017.json",
        "instances_val2017.json",
        "instances_test2017.json",
    ]

    missing = [split for split in split_files if not (args.annotations_dir / split).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing annotation files in {args.annotations_dir}: {', '.join(missing)}"
        )

    for split in split_files:
        path = args.annotations_dir / split
        results = analyze_split(path)
        print_results(split.replace("instances_", ""), results)


if __name__ == "__main__":
    main()
