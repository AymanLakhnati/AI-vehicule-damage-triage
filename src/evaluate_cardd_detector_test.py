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

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / 'data' / 'raw' / 'cardd' / 'CarDD_release' / 'CarDD_COCO'
MODELS_DIR = REPO_ROOT / 'models'

# Test-specific paths
TEST_ANNOTATIONS_PATH = (DATA_ROOT / 'annotations' / 'instances_test2017.json')
TEST_IMAGES_DIR = DATA_ROOT / 'test2017'
CHECKPOINT_PATH = Path(
    '/content/drive/MyDrive/vehicle-damage-triage-models/'
    'cardd_detector_epoch3.pth'
)

REPORT_PATH = REPO_ROOT / 'reports' / 'cardd_detector_test_results.txt'
BATCH_SIZE = 2
NUM_WORKERS = 0


def collate_fn(batch):
    return tuple(zip(*batch))


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_detector().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_test_loader() -> DataLoader:
    dataset = CarDDDetectionDataset(str(TEST_ANNOTATIONS_PATH), str(TEST_IMAGES_DIR))
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
        f'Test mAP@[0.50:0.95]: {results["mAP"]:.6f}',
        f'Test mAP@50: {results["mAP_50"]:.6f}',
        f'Test mAP@75: {results["mAP_75"]:.6f}',
        f'Test AR@100: {results["mAR_100"]:.6f}',
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

    print(f'Loading checkpoint: {CHECKPOINT_PATH}')
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f'Checkpoint not found: {CHECKPOINT_PATH}')

    model = load_model(CHECKPOINT_PATH, device)
    test_loader = build_test_loader()

    results = evaluate(model, test_loader, device)

    print(f'Test mAP@[0.50:0.95]: {results["mAP"]:.6f}')
    print(f'Test mAP@50: {results["mAP_50"]:.6f}')
    print(f'Test mAP@75: {results["mAP_75"]:.6f}')
    print(f'Test AR@100: {results["mAR_100"]:.6f}')

    save_results(results)
    print(f'Results saved to: {REPORT_PATH}')


if __name__ == '__main__':
    main()
