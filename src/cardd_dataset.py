import json
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


CLASSES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]

COCO_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


class CarDDMultiLabelDataset(Dataset):
    def __init__(self, annotations_json, images_dir, transform=None):
        self.annotations_json = Path(annotations_json)
        self.images_dir = Path(images_dir)
        self.transform = transform or T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ])

        if not self.annotations_json.exists():
            raise FileNotFoundError(f"Annotations JSON not found: {self.annotations_json}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")

        with open(self.annotations_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images = data.get('images', [])
        annotations = data.get('annotations', [])

        # Map image_id -> file_name
        self.image_id_to_file = {img['id']: img['file_name'] for img in images}
        self.image_ids = list(self.image_id_to_file.keys())

        # Map image_id -> set(category_ids)
        self.image_to_categories = defaultdict(set)
        for ann in annotations:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            self.image_to_categories[img_id].add(cat_id)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        file_name = self.image_id_to_file[img_id]
        img_path = self.images_dir / file_name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        with Image.open(img_path) as img:
            img = img.convert('RGB')
            image_tensor = self.transform(img)

        target = torch.zeros(len(CLASSES), dtype=torch.float32)
        for cat_id in self.image_to_categories.get(img_id, []):
            idx_map = COCO_TO_INDEX.get(cat_id)
            if idx_map is not None:
                target[idx_map] = 1.0

        return image_tensor, target


if __name__ == '__main__':
    # Quick test for train2017
    ann = Path('data/raw/cardd/CarDD_release/CarDD_COCO/annotations/instances_train2017.json')
    imgs = Path('data/raw/cardd/CarDD_release/CarDD_COCO/train2017')
    dataset = CarDDMultiLabelDataset(ann, imgs)
    print(f"Dataset size: {len(dataset)}")
    image, target = dataset[0]
    print("Image shape:", image.shape)
    print("Target:", target)
    print("Target shape:", target.shape)
