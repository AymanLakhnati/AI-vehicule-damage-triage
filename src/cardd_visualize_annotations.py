import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


CATEGORY_ORDER = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]

CATEGORY_COLORS = {
    "dent": "tab:blue",
    "scratch": "tab:orange",
    "crack": "tab:green",
    "glass shatter": "tab:red",
    "lamp broken": "tab:purple",
    "tire flat": "tab:brown",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_annotation_index(annotations, category_map):
    index = defaultdict(list)
    for ann in annotations:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        category_name = category_map.get(category_id, str(category_id))
        index[image_id].append({
            "bbox": ann["bbox"],
            "category": category_name,
        })
    return index


def select_images(images, num_images, seed=None):
    if seed is not None:
        random.seed(seed)
    image_ids = [img["id"] for img in images]
    selected_ids = random.sample(image_ids, min(num_images, len(image_ids)))
    image_map = {img["id"]: img for img in images}
    return [image_map[i] for i in selected_ids]


def plot_image(ax, image_path, ann_list):
    image = plt.imread(str(image_path))
    ax.imshow(image)
    ax.axis("off")

    for ann in ann_list:
        x, y, width, height = ann["bbox"]
        category = ann["category"]
        color = CATEGORY_COLORS.get(category, "yellow")
        rect = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y - 4,
            category,
            fontsize=8,
            color=color,
            backgroundcolor="black",
            alpha=0.65,
        )

    title = image_path.name
    ax.set_title(title, fontsize=9)


def main():
    parser = argparse.ArgumentParser(description="Visualize CarDD train annotations with bounding boxes")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/raw/cardd/CarDD_release/CarDD_COCO/annotations"),
        help="Directory containing CarDD COCO annotation JSON files",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/raw/cardd/CarDD_release/CarDD_COCO/train2017"),
        help="Directory containing CarDD train images",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of random training images to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the visualization image",
    )
    args = parser.parse_args()

    annotations_path = args.annotations_dir / "instances_train2017.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotations_path}")

    data = load_json(annotations_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    if not images:
        raise ValueError("No images found in annotation file.")
    if not annotations:
        raise ValueError("No annotations found in annotation file.")

    category_map = {cat["id"]: cat["name"] for cat in categories}
    annotation_index = build_annotation_index(annotations, category_map)
    selected_images = select_images(images, args.num_images, seed=args.seed)

    cols = 4
    rows = (len(selected_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, image_info in zip(axes, selected_images):
        image_filename = image_info.get("file_name")
        image_path = args.images_dir / image_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        ann_list = annotation_index.get(image_info["id"], [])
        plot_image(ax, image_path, ann_list)

    for ax in axes[len(selected_images):]:
        ax.axis("off")

    fig.suptitle("CarDD train annotations", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(args.save), dpi=150)
        print(f"Saved visualization to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
