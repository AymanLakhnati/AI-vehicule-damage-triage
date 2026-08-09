from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError as exc:
    raise ImportError(
        'torchmetrics is required for evaluation. Install it with: py -m pip install torchmetrics'
    ) from exc

from cardd_detection_dataset import CarDDDetectionDataset
from cardd_detector import build_detector

MODELS_DIR = Path('models')
VAL_ANNOTATIONS_PATH = Path('data/raw/cardd/CarDD_release/CarDD_COCO/annotations/instances_val2017.json')
VAL_IMAGES_DIR = Path('data/raw/cardd/CarDD_release/CarDD_COCO/val2017')
REPORT_PATH = Path('reports/cardd_detector_validation_results.txt')
BATCH_SIZE = 2
NUM_WORKERS = 0


def collate_fn(batch):
    return tuple(zip(*batch))


def find_latest_checkpoint(model_dir: Path) -> Path:
    if not model_dir.exists():
        raise FileNotFoundError(f'Model directory not found: {model_dir}')

    checkpoint_paths = sorted(model_dir.glob('cardd_detector_epoch*.pth'))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f'No checkpoint files found in {model_dir}. Run training first to produce a checkpoint.'
        )
    return checkpoint_paths[-1]


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_detector().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_val_loader() -> DataLoader:
    dataset = CarDDDetectionDataset(str(VAL_ANNOTATIONS_PATH), str(VAL_IMAGES_DIR))
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )


def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> dict:
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        class_metrics=True,
    )

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            predictions = [
                {
                    'boxes': out['boxes'].cpu(),
                    'scores': out['scores'].cpu(),
                    'labels': out['labels'].cpu(),
                }
                for out in outputs
            ]
            reference_targets = [
                {
                    'boxes': tgt['boxes'].cpu(),
                    'labels': tgt['labels'].cpu(),
                }
                for tgt in targets
            ]

            metric.update(predictions, reference_targets)

    results = metric.compute()
    return {
        'mAP': results['map'].item(),
        'mAP_50': results['map_50'].item(),
        'mAP_75': results['map_75'].item(),
        'mAR_100': results['mar_100'].item(),
        'per_class_AP': results.get('map_per_class'),
        'per_class_AR': results.get('mar_100_per_class'),
    }


def save_results(results: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'mAP: {results["mAP"]:.6f}',
        f'mAP@50: {results["mAP_50"]:.6f}',
        f'mAP@75: {results["mAP_75"]:.6f}',
        f'mAR@100: {results["mAR_100"]:.6f}',
        '',
    ]

    if results['per_class_AP'] is not None:
        lines.append('per_class_AP:')
        for idx, value in enumerate(results['per_class_AP']):
            lines.append(f'  class_{idx + 1}: {value:.6f}')
        lines.append('')

    if results['per_class_AR'] is not None:
        lines.append('per_class_AR@100:')
        for idx, value in enumerate(results['per_class_AR']):
            lines.append(f'  class_{idx + 1}: {value:.6f}')
        lines.append('')

    REPORT_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    checkpoint_path = find_latest_checkpoint(MODELS_DIR)
    print(f'Loading checkpoint: {checkpoint_path}')
    model = load_model(checkpoint_path, device)
    val_loader = build_val_loader()

    results = evaluate(model, val_loader, device)

    print(f'mAP: {results["mAP"]:.6f}')
    print(f'mAP@50: {results["mAP_50"]:.6f}')
    print(f'mAP@75: {results["mAP_75"]:.6f}')
    print(f'mAR@100: {results["mAR_100"]:.6f}')

    save_results(results)
    print(f'Results saved to: {REPORT_PATH}')


if __name__ == '__main__':
    main()
