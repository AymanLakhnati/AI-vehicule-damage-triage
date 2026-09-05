import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from cardd_detection_dataset import CarDDDetectionDataset
from cardd_detector import build_detector

CLASS_NAMES = {
    1: "dent",
    2: "scratch",
    3: "crack",
    4: "glass shatter",
    5: "lamp broken",
    6: "tire flat",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "cardd" / "CarDD_release" / "CarDD_COCO"
DEFAULT_ANNOTATIONS = DATA_ROOT / "annotations" / "instances_test2017.json"
DEFAULT_IMAGES = DATA_ROOT / "test2017"
DEFAULT_THRESHOLDS = REPO_ROOT / "models" / "cardd_detector_thresholds.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "cardd_detector_thresholded_results.txt"


def collate_fn(batch):
    return tuple(zip(*batch))


def load_thresholds(path):
    with path.open("r", encoding="utf-8") as file:
        values = json.load(file)
    return {class_id: float(values[CLASS_NAMES[class_id]]) for class_id in CLASS_NAMES}


def load_model(checkpoint_path, device):
    model = build_detector().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def update_counts(counts, prediction, target, thresholds, iou_threshold):
    pred_scores = prediction["scores"].cpu()
    keep = torch.tensor([
        score >= thresholds.get(int(label), 1.0)
        for score, label in zip(pred_scores, prediction["labels"].cpu())
    ], dtype=torch.bool)
    pred_boxes = prediction["boxes"].cpu()[keep]
    pred_labels = prediction["labels"].cpu()[keep]
    pred_scores = pred_scores[keep]
    true_boxes = target["boxes"].cpu()
    true_labels = target["labels"].cpu()
    matched = set()
    ious = box_iou(pred_boxes, true_boxes) if len(pred_boxes) and len(true_boxes) else None

    for pred_index in torch.argsort(pred_scores, descending=True).tolist():
        label = int(pred_labels[pred_index])
        if label not in counts:
            continue
        candidates = [
            true_index
            for true_index in range(len(true_boxes))
            if int(true_labels[true_index]) == label and true_index not in matched
        ]
        best = max(candidates, key=lambda index: float(ious[pred_index, index])) if candidates and ious is not None else None
        if best is not None and float(ious[pred_index, best]) >= iou_threshold:
            matched.add(best)
            counts[label]["tp"] += 1
        else:
            counts[label]["fp"] += 1

    for true_index, label in enumerate(true_labels.tolist()):
        if true_index not in matched and int(label) in counts:
            counts[int(label)]["fn"] += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a detector with validation-tuned class thresholds on untouched test data.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    return parser.parse_args()


def main():
    args = parse_args()
    for path, name in (
        (args.checkpoint, "checkpoint"),
        (args.thresholds, "threshold file"),
        (args.annotations, "annotations"),
        (args.images, "image directory"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name.capitalize()} not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    thresholds = load_thresholds(args.thresholds)
    print("Frozen thresholds:", {CLASS_NAMES[key]: value for key, value in thresholds.items()})

    model = load_model(args.checkpoint, device)
    dataset = CarDDDetectionDataset(str(args.annotations), str(args.images))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in CLASS_NAMES}

    with torch.no_grad():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                update_counts(counts, output, target, thresholds, args.iou_threshold)

    lines = [
        "Thresholded detector evaluation on untouched test2017",
        f"Checkpoint: {args.checkpoint}",
        f"Thresholds: {args.thresholds}",
        f"IoU threshold: {args.iou_threshold:.2f}",
        "",
        "class              TP      FP      FN     precision   recall       F1",
        "-" * 76,
    ]
    for class_id, name in CLASS_NAMES.items():
        value = counts[class_id]
        precision = value["tp"] / (value["tp"] + value["fp"]) if value["tp"] + value["fp"] else 0.0
        recall = value["tp"] / (value["tp"] + value["fn"]) if value["tp"] + value["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lines.append(f"{name:18} {value['tp']:6d} {value['fp']:7d} {value['fn']:7d} {precision:11.4f} {recall:8.4f} {f1:9.4f}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport saved to: {args.report}")


if __name__ == "__main__":
    main()
