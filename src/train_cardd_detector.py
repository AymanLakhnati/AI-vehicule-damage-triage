import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cardd_detection_dataset import CarDDDetectionDataset
from cardd_detector import build_detector

TRAIN_ANNOTATIONS_PATH = 'data/raw/cardd/CarDD_release/CarDD_COCO/annotations/instances_train2017.json'
VAL_ANNOTATIONS_PATH = 'data/raw/cardd/CarDD_release/CarDD_COCO/annotations/instances_val2017.json'
TRAIN_IMAGES_DIR = 'data/raw/cardd/CarDD_release/CarDD_COCO/train2017'
VAL_IMAGES_DIR = 'data/raw/cardd/CarDD_release/CarDD_COCO/val2017'
MODELS_DIR = Path('models')
EPOCHS = 5
BATCH_SIZE = 2
NUM_WORKERS = 0

LABEL_TO_COCO = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
}


def collate_fn(batch):
    return tuple(zip(*batch))


def save_checkpoint(model: torch.nn.Module, epoch: int):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f'cardd_detector_epoch{epoch}.pth'
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')


def evaluate_coco(model: torch.nn.Module, data_loader: DataLoader, annotations_file: str, device: torch.device):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print('pycocotools is not installed. Skipping COCO evaluation.')
        return {}

    model.eval()
    coco_gt = COCO(annotations_file)
    detections = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id = int(target['image_id'].item())
                boxes = output['boxes'].cpu().tolist()
                scores = output['scores'].cpu().tolist()
                labels = output['labels'].cpu().tolist()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    detections.append({
                        'image_id': image_id,
                        'category_id': LABEL_TO_COCO[label],
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': float(score),
                    })

    if not detections:
        print('No detections generated for evaluation.')
        return {}

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        'mAP': float(stats[0]),
        'mAP_50': float(stats[1]),
        'mAP_75': float(stats[2]),
        'mAP_small': float(stats[3]),
        'mAP_medium': float(stats[4]),
        'mAP_large': float(stats[5]),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset = CarDDDetectionDataset(TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
    val_dataset = CarDDDetectionDataset(VAL_ANNOTATIONS_PATH, VAL_IMAGES_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model = build_detector().to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_losses = []
        loss_components = {
            'loss_classifier': [],
            'loss_box_reg': [],
            'loss_objectness': [],
            'loss_rpn_box_reg': [],
        }

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_losses.append(loss.item())
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key].append(loss_dict[key].item())

        avg_total = sum(total_losses) / len(total_losses)
        avg_classifier = sum(loss_components['loss_classifier']) / len(loss_components['loss_classifier'])
        avg_box_reg = sum(loss_components['loss_box_reg']) / len(loss_components['loss_box_reg'])
        avg_objectness = sum(loss_components['loss_objectness']) / len(loss_components['loss_objectness'])
        avg_rpn_box = sum(loss_components['loss_rpn_box_reg']) / len(loss_components['loss_rpn_box_reg'])

        print(f'Epoch {epoch}/{EPOCHS}')
        print(f'total_loss={avg_total:.4f}')
        print(f'classifier_loss={avg_classifier:.4f}')
        print(f'box_reg_loss={avg_box_reg:.4f}')
        print(f'objectness_loss={avg_objectness:.4f}')
        print(f'rpn_box_loss={avg_rpn_box:.4f}')

        save_checkpoint(model, epoch)

        eval_metrics = evaluate_coco(model, val_loader, VAL_ANNOTATIONS_PATH, device)
        if eval_metrics:
            print(f"val_mAP={eval_metrics['mAP']:.4f}")
            print(f"val_mAP_50={eval_metrics['mAP_50']:.4f}")
            print(f"val_mAP_75={eval_metrics['mAP_75']:.4f}")
        print('')


if __name__ == '__main__':
    main()
