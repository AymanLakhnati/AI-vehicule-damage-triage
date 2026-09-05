import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
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
DEFAULT_REPORT = REPO_ROOT / "reports" / "cardd_detector_error_analysis.txt"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "detector_error_analysis"


def collate_fn(batch):
    return tuple(zip(*batch))


def load_model(checkpoint_path, device):
    model = build_detector().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def match_detections(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, iou_threshold):
    matched_true = set()
    true_positives = []
    false_positives = []

    order = torch.argsort(pred_scores, descending=True)
    ious = box_iou(pred_boxes, true_boxes) if len(pred_boxes) and len(true_boxes) else None

    for pred_index in order.tolist():
        label = int(pred_labels[pred_index])
        candidates = [
            true_index
            for true_index in range(len(true_boxes))
            if int(true_labels[true_index]) == label and true_index not in matched_true
        ]
        best_iou = 0.0
        best_true_index = None
        if ious is not None and candidates:
            best_true_index = max(candidates, key=lambda index: float(ious[pred_index, index]))
            best_iou = float(ious[pred_index, best_true_index])

        prediction = {
            "box": pred_boxes[pred_index],
            "label": label,
            "score": float(pred_scores[pred_index]),
        }
        if best_true_index is not None and best_iou >= iou_threshold:
            matched_true.add(best_true_index)
            true_positives.append(prediction)
        else:
            false_positives.append(prediction)

    false_negatives = [
        {"box": true_boxes[index], "label": int(true_labels[index])}
        for index in range(len(true_boxes))
        if index not in matched_true
    ]
    return true_positives, false_positives, false_negatives


def draw_examples(image_path, output_path, true_positives, false_positives, false_negatives):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)

    for item in true_positives:
        draw.rectangle(item["box"].tolist(), outline=(0, 180, 0), width=3)
    for item in false_positives:
        draw.rectangle(item["box"].tolist(), outline=(220, 0, 0), width=3)
        x1, y1, _, _ = item["box"].tolist()
        draw.text((x1, max(0, y1 - 16)), f"FP {CLASS_NAMES[item['label']]} {item['score']:.2f}", fill=(220, 0, 0))
    for item in false_negatives:
        draw.rectangle(item["box"].tolist(), outline=(255, 160, 0), width=3)
        x1, y1, _, _ = item["box"].tolist()
        draw.text((x1, max(0, y1 - 16)), f"FN {CLASS_NAMES[item['label']]}", fill=(255, 160, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze detector errors on the untouched CarDD test split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Epoch 3 detector checkpoint path.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Minimum score to include in error analysis.")
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="IoU required for a true positive.")
    parser.add_argument("--examples-per-class", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    for path, description in (
        (args.checkpoint, "checkpoint"),
        (args.annotations, "annotations"),
        (args.images, "image directory"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{description.capitalize()} not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    dataset = CarDDDetectionDataset(str(args.annotations), str(args.images))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in CLASS_NAMES}
    examples = {class_id: [] for class_id in CLASS_NAMES}

    with torch.no_grad():
        for images, targets in loader:
            output = model([images[0].to(device)])[0]
            keep = output["scores"].cpu() >= args.score_threshold
            pred_boxes = output["boxes"].cpu()[keep]
            pred_labels = output["labels"].cpu()[keep]
            pred_scores = output["scores"].cpu()[keep]
            target = targets[0]
            tp, fp, fn = match_detections(
                pred_boxes,
                pred_labels,
                pred_scores,
                target["boxes"],
                target["labels"],
                args.iou_threshold,
            )

            for item in tp:
                counts[item["label"]]["tp"] += 1
            for item in fp:
                counts[item["label"]]["fp"] += 1
            for item in fn:
                counts[item["label"]]["fn"] += 1

            image_id = int(target["image_id"].item())
            image_name = dataset.image_id_to_file[image_id]
            image_path = args.images / image_name
            involved = {item["label"] for item in tp + fp + fn}
            for class_id in involved:
                class_errors = [item for item in fp + fn if item["label"] == class_id]
                if class_errors and len(examples[class_id]) < args.examples_per_class:
                    output_path = args.output / CLASS_NAMES[class_id].replace(" ", "_") / f"{len(examples[class_id]) + 1:03d}_{Path(image_name).stem}.jpg"
                    draw_examples(image_path, output_path, [item for item in tp if item["label"] == class_id], [item for item in fp if item["label"] == class_id], [item for item in fn if item["label"] == class_id])
                    examples[class_id].append(str(output_path))

    lines = [
        "Detector error analysis on untouched test2017",
        f"Checkpoint: {args.checkpoint}",
        f"Score threshold: {args.score_threshold:.2f}",
        f"IoU threshold: {args.iou_threshold:.2f}",
        "",
        "Per-class counts:",
        "class              TP      FP      FN     precision   recall",
        "-" * 65,
    ]
    for class_id, name in CLASS_NAMES.items():
        values = counts[class_id]
        precision = values["tp"] / (values["tp"] + values["fp"]) if values["tp"] + values["fp"] else 0.0
        recall = values["tp"] / (values["tp"] + values["fn"]) if values["tp"] + values["fn"] else 0.0
        lines.append(f"{name:18} {values['tp']:6d} {values['fp']:7d} {values['fn']:7d} {precision:11.4f} {recall:8.4f}")

    lines.extend(["", "Saved examples:"])
    for class_id, paths in examples.items():
        lines.append(f"{CLASS_NAMES[class_id]}:")
        lines.extend(f"  {path}" for path in paths)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport saved to: {args.report}")


if __name__ == "__main__":
    main()
