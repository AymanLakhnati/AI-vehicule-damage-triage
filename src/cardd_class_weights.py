import json
from pathlib import Path

import torch


CLASSES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]

# COCO category ids -> index in CLASSES
COCO_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def compute_pos_weights(annotations_json_path: Path):
    with annotations_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    image_ids = [img["id"] for img in images]
    total_images = len(image_ids)

    # Map image_id -> set of category_ids present in that image
    image_to_cats = {img_id: set() for img_id in image_ids}
    for ann in annotations:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id in image_to_cats:
            image_to_cats[img_id].add(cat_id)

    positives = [0] * len(CLASSES)
    for img_id, cats in image_to_cats.items():
        for cat_id in cats:
            idx = COCO_TO_INDEX.get(cat_id)
            if idx is not None:
                positives[idx] += 1

    negatives = [total_images - p for p in positives]

    pos_weights = []
    for cls, p, n in zip(CLASSES, positives, negatives):
        if p > 0:
            pw = n / p
        else:
            pw = float('inf')
        print(f"{cls}: positives={p}, negatives={n}, pos_weight={pw:.6f}")
        pos_weights.append(pw if p > 0 else 0.0)

    tensor = torch.tensor(pos_weights, dtype=torch.float32)
    print('\npos_weights tensor:', tensor)
    return tensor


if __name__ == '__main__':
    ann = Path('data/raw/cardd/CarDD_release/CarDD_COCO/annotations/instances_train2017.json')
    if not ann.exists():
        raise FileNotFoundError(f"Training annotation not found: {ann}")
    compute_pos_weights(ann)
