import argparse
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

# Class names for vehicle damage detection
CLASS_NAMES = {
    1: "dent",
    2: "scratch",
    3: "crack",
    4: "glass shatter",
    5: "lamp broken",
    6: "tire flat",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / 'data' / 'raw' / 'cardd' / 'CarDD_release' / 'CarDD_COCO'
MODELS_DIR = REPO_ROOT / 'models'

# Test-specific paths
TEST_ANNOTATIONS_PATH = (DATA_ROOT / 'annotations' / 'instances_test2017.json')
TEST_IMAGES_DIR = DATA_ROOT / 'test2017'
REPORT_PATH = REPO_ROOT / 'reports' / 'cardd_detector_per_class_results.txt'
BATCH_SIZE = 2
NUM_WORKERS = 0


def find_latest_checkpoint(model_dir: Path) -> Path:
    """Find the latest detector checkpoint (Epoch N)."""
    checkpoint_paths = sorted(model_dir.glob('cardd_detector_epoch*.pth'))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f'No checkpoint files found in {model_dir}. Run training first to produce a checkpoint, '
            'or pass --checkpoint /path/to/cardd_detector_epoch3.pth.'
        )
    return checkpoint_paths[-1]


def resolve_checkpoint(checkpoint_arg: Path | None) -> Path:
    if checkpoint_arg is not None:
        checkpoint_path = checkpoint_arg.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        return checkpoint_path

    try:
        return find_latest_checkpoint(MODELS_DIR)
    except FileNotFoundError:
        fallback = REPO_ROOT / 'models' / 'cardd_detector_epoch3.pth'
        if fallback.exists():
            return fallback
        raise


def collate_fn(batch):
    return tuple(zip(*batch))


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the detector model from checkpoint."""
    model = build_detector().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_test_loader() -> DataLoader:
    """Build test data loader."""
    dataset = CarDDDetectionDataset(str(TEST_ANNOTATIONS_PATH), str(TEST_IMAGES_DIR))
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )


def evaluate_per_class(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate model per class and return per-class AP metrics.
    """
    # Overall metric to get per-class stats
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

    # Extract per-class AP
    per_class_ap = results.get('map_per_class')
    
    return {
        'mAP': results['map'].item(),
        'mAP_50': results['map_50'].item(),
        'mAP_75': results['map_75'].item(),
        'mAR_100': results['mar_100'].item(),
        'per_class_AP': per_class_ap,
    }


def save_results(results: dict, class_names: dict) -> None:
    """Save per-class results to file."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        'Per-class test performance (Epoch 3)',
        '=' * 50,
        '',
    ]

    # Overall metrics
    lines.append(f'Overall mAP@[0.50:0.95]: {results["mAP"]:.6f}')
    lines.append(f'Overall mAP@50: {results["mAP_50"]:.6f}')
    lines.append(f'Overall mAP@75: {results["mAP_75"]:.6f}')
    lines.append(f'Overall mAR@100: {results["mAR_100"]:.6f}')
    lines.append('')
    lines.append('Per-class AP:')
    lines.append('-' * 50)

    # Per-class AP
    per_class_ap = results['per_class_AP']
    if per_class_ap is not None:
        for class_id in sorted(class_names.keys()):
            class_name = class_names[class_id]
            ap_value = per_class_ap[class_id - 1].item()
            lines.append(f'{class_name:18} AP = {ap_value:.6f}')
    else:
        lines.append('(Per-class metrics not available)')

    REPORT_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the detector per damage class on the test split.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help='Path to the detector checkpoint (.pth). Defaults to the latest model in models/.',
    )
    parser.add_argument(
        '--annotations',
        type=Path,
        default=TEST_ANNOTATIONS_PATH,
        help='Path to instances_test2017.json.',
    )
    parser.add_argument(
        '--images',
        type=Path,
        default=TEST_IMAGES_DIR,
        help='Path to the test image directory.',
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=REPORT_PATH,
        help='Output report path.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    global REPORT_PATH
    REPORT_PATH = args.report

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Loading checkpoint: {checkpoint_path}')

    model = load_model(checkpoint_path, device)
    test_loader = DataLoader(
        CarDDDetectionDataset(str(args.annotations), str(args.images)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    print('Evaluating on test set...')
    results = evaluate_per_class(model, test_loader, device)

    # Print results
    print('\nPer-class test performance')
    print('=' * 50)
    print(f'Overall mAP@[0.50:0.95]: {results["mAP"]:.6f}')
    print(f'Overall mAP@50: {results["mAP_50"]:.6f}')
    print(f'Overall mAP@75: {results["mAP_75"]:.6f}')
    print(f'Overall mAR@100: {results["mAR_100"]:.6f}')
    print('')
    print('Per-class AP:')
    print('-' * 50)

    per_class_ap = results['per_class_AP']
    if per_class_ap is not None:
        for class_id in sorted(CLASS_NAMES.keys()):
            class_name = CLASS_NAMES[class_id]
            ap_value = per_class_ap[class_id - 1].item()
            print(f'{class_name:18} AP = {ap_value:.6f}')
    else:
        print('(Per-class metrics not available)')

    save_results(results, CLASS_NAMES)
    print(f'\nResults saved to: {REPORT_PATH}')


if __name__ == '__main__':
    main()
