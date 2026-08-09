import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from cardd_detection_dataset import CarDDDetectionDataset, CLASSES

ANNOTATIONS_PATH = Path('data/raw/cardd/CarDD_release/CarDD_COCO/annotations/instances_train2017.json')
IMAGES_DIR = Path('data/raw/cardd/CarDD_release/CarDD_COCO/train2017')


def draw_sample(image_tensor, target, ax):
    image = image_tensor.permute(1, 2, 0).numpy()
    image = (image * 255).astype('uint8')
    ax.imshow(image)
    ax.axis('off')

    boxes = target['boxes'].cpu().numpy()
    labels = target['labels'].cpu().numpy()

    for box, label_idx in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        label_text = CLASSES[label_idx - 1] if 0 <= label_idx - 1 < len(CLASSES) else str(label_idx)
        ax.text(x1, y1 - 5, label_text, color='yellow', fontsize=10, weight='bold', backgroundcolor='black')


def main():
    dataset = CarDDDetectionDataset(ANNOTATIONS_PATH, IMAGES_DIR)
    print(f"Dataset size: {len(dataset)}")

    num_samples = min(8, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(num_samples // 2, 2, figsize=(14, num_samples * 2.5))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        image, target = dataset[idx]
        print('Sample', idx)
        print('  Image shape:', image.shape)
        print('  Boxes shape:', target['boxes'].shape)
        print('  Labels shape:', target['labels'].shape)
        draw_sample(image, target, ax)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
