import torch
from torch.utils.data import DataLoader

from cardd_detection_dataset import CarDDDetectionDataset
from cardd_detector import build_detector

ANNOTATIONS_PATH = 'data/raw/cardd/CarDD_release/CarDD_COCO/annotations/instances_train2017.json'
IMAGES_DIR = 'data/raw/cardd/CarDD_release/CarDD_COCO/train2017'


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    dataset = CarDDDetectionDataset(ANNOTATIONS_PATH, IMAGES_DIR)
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    images, targets = next(iter(train_loader))

    print('Number of images:', len(images))
    print('Image 0 shape:', images[0].shape)

    print('Number of targets:', len(targets))
    print('Boxes 0:', targets[0]['boxes'].shape)
    print('Labels 0:', targets[0]['labels'].shape)

    model = build_detector()
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    images = [img.to(device) for img in images]
    targets = [{
        'boxes': t['boxes'].to(device),
        'labels': t['labels'].to(device),
        'image_id': t['image_id'].to(device),
        'area': t['area'].to(device),
        'iscrowd': t['iscrowd'].to(device),
    } for t in targets]

    loss_dict = model(images, targets)

    print('Loss dict:', loss_dict)
    print('Total loss:', sum(loss_dict.values()).item())


if __name__ == '__main__':
    main()
