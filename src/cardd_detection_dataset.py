import json
from pathlib import Path

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

COCO_TO_INDEX = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
}

DEFAULT_TRANSFORM = T.Compose([
    T.ToTensor(),
])


class CarDDDetectionDataset(Dataset):
    def __init__(self, annotations_json, images_dir, transform=None):
        self.annotations_json = Path(annotations_json)
        self.images_dir = Path(images_dir)
        self.transform = transform or DEFAULT_TRANSFORM

        if not self.annotations_json.exists():
            raise FileNotFoundError(f"Annotations JSON not found: {self.annotations_json}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")

        with open(self.annotations_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images = data.get('images', [])
        annotations = data.get('annotations', [])

        self.image_id_to_file = {img['id']: img['file_name'] for img in images}
        self.image_ids = list(self.image_id_to_file.keys())

        self.image_to_annotations = {img_id: [] for img_id in self.image_ids}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                continue
            self.image_to_annotations[img_id].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = self.image_id_to_file[image_id]
        image_path = self.images_dir / file_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            img = img.convert('RGB')
            image = self.transform(img)

        annotations = self.image_to_annotations.get(image_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            bbox = ann.get('bbox', [])
            if len(bbox) != 4:
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(COCO_TO_INDEX.get(ann['category_id'], -1) + 1)
            areas.append(w * h)
            iscrowd.append(ann.get('iscrowd', 0))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': areas,
            'iscrowd': iscrowd,
        }

        return image, target


if __name__ == '__main__':
    ann = Path('data/raw/cardd/CarDD_release/CarDD_COCO/annotations/instances_train2017.json')
    imgs = Path('data/raw/cardd/CarDD_release/CarDD_COCO/train2017')
    dataset = CarDDDetectionDataset(ann, imgs)
    print(f"Dataset size: {len(dataset)}")
    image, target = dataset[0]
    print('Image shape:', image.shape)
    print('Target keys:', target.keys())
    print('Boxes shape:', target['boxes'].shape)
    print('Labels shape:', target['labels'].shape)
    print('Image ID:', target['image_id'])
