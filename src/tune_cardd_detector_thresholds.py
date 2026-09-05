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
DEFAULT_ANNOTATIONS = DATA_ROOT / "annotations" / "instances_val2017.json"
DEFAULT_IMAGES = DATA_ROOT / "val2017"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "cardd_detector_thresholds.json"
THRESHOLDS = [round(value, 2) for value in torch.arange(0.05, 1.00, 0.05).tolist()]


def collate_fn(batch):
    return tuple(zip(*batch))


def load_model(checkpoint_path, device):
    model = build_detector().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def match_at_threshold(prediction, target, score_threshold, iou_threshold):
    keep = prediction["scores"] >= score_threshold
    pred_boxes = prediction["boxes"][keep]
    pred_labels = prediction["labels"][keep]
    pred_scores = prediction["scores"][keep]
    true_boxes = target["boxes"]
    true_labels = target["labels"]
    matched = set()
    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in CLASS_NAMES}

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
    return counts


def collect_predictions(model, loader, device):
    collected = []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                collected.append(
                    (
                        {key: value.detach().cpu() for key, value in output.items()},
                        {key: value.detach().cpu() for key, value in target.items()},
                    )
                )
    return collected


def score_threshold(predictions, class_id, threshold, iou_threshold):
    tp = fp = fn = 0
    for prediction, target in predictions:
        counts = match_at_threshold(prediction, target, threshold, iou_threshold)[class_id]
        tp += counts["tp"]
        fp += counts["fp"]
        fn += counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def parse_args():
    parser = argparse.ArgumentParser(description="Tune detector confidence thresholds on CarDD validation data.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Epoch 3 detector checkpoint.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    return parser.parse_args()


def main():
    args = parse_args()
    for path, name in ((args.checkpoint, "checkpoint"), (args.annotations, "annotations"), (args.images, "images")):
        if not path.exists():
            raise FileNotFoundError(f"{name.capitalize()} not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    dataset = CarDDDetectionDataset(str(args.annotations), str(args.images))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"Collecting validation predictions from {len(dataset)} images...")
    predictions = collect_predictions(model, loader, device)

    selected = {}
    for class_id, class_name in CLASS_NAMES.items():
        scores = [score_threshold(predictions, class_id, threshold, args.iou_threshold) for threshold in THRESHOLDS]
        best = max(scores, key=lambda result: (result["f1"], result["recall"], -result["threshold"]))
        selected[class_name] = best["threshold"]
        print(
            f"{class_name:18} threshold={best['threshold']:.2f} "
            f"precision={best['precision']:.4f} recall={best['recall']:.4f} F1={best['f1']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(f"Saved validation-tuned thresholds to: {args.output}")


if __name__ == "__main__":
    main()
