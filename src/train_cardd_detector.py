import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cardd_detection_dataset import CarDDDetectionDataset
from cardd_detector import build_detector


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    REPO_ROOT
    / "data"
    / "raw"
    / "cardd"
    / "CarDD_release"
    / "CarDD_COCO"
)

TRAIN_ANNOTATIONS_PATH = (
    DATA_ROOT
    / "annotations"
    / "instances_train2017.json"
)

VAL_ANNOTATIONS_PATH = (
    DATA_ROOT
    / "annotations"
    / "instances_val2017.json"
)

TRAIN_IMAGES_DIR = DATA_ROOT / "train2017"
VAL_IMAGES_DIR = DATA_ROOT / "val2017"


# ============================================================
# Checkpoint directory
# ============================================================

DRIVE_MODELS_DIR = Path(
    "/content/drive/MyDrive/vehicle-damage-triage-models"
)

# If Google Drive is mounted in Colab, save there permanently.
# Otherwise, save to the local project models/ folder.
if Path("/content/drive/MyDrive").exists():
    MODELS_DIR = DRIVE_MODELS_DIR
else:
    MODELS_DIR = REPO_ROOT / "models"


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 2
NUM_WORKERS = 0


# Detector labels and COCO category IDs are identical for CarDD.
LABEL_TO_COCO = {
    1: 1,  # dent
    2: 2,  # scratch
    3: 3,  # crack
    4: 4,  # glass shatter
    5: 5,  # lamp broken
    6: 6,  # tire flat
}


# ============================================================
# DataLoader helper
# ============================================================

def collate_fn(batch):
    """
    Object-detection datasets return images and targets with
    potentially different shapes and numbers of bounding boxes.

    Therefore, we cannot stack them like normal classification
    batches.
    """
    return tuple(zip(*batch))


# ============================================================
# Checkpoint saving
# ============================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
):
    """
    Save both:

    1. A normal model state_dict for evaluation/inference.
    2. A training-state checkpoint containing the optimizer
       so training can later be resumed.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = (
        MODELS_DIR
        / f"cardd_detector_epoch{epoch}.pth"
    )

    training_path = (
        MODELS_DIR
        / f"cardd_detector_epoch{epoch}_training.pth"
    )

    # Normal model checkpoint
    torch.save(
        model.state_dict(),
        model_path,
    )

    # Training-state checkpoint
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        training_path,
    )

    print(f"Saved model permanently to: {model_path}")
    print(f"Saved training state to: {training_path}")


# ============================================================
# COCO evaluation
# ============================================================

def evaluate_coco(
    model: torch.nn.Module,
    data_loader: DataLoader,
    annotations_file: Path,
    device: torch.device,
):
    """
    Evaluate Faster R-CNN using standard COCO bounding-box metrics.
    """

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

    except ImportError:
        print(
            "pycocotools is not installed. "
            "Skipping COCO evaluation."
        )
        return {}

    model.eval()

    coco_gt = COCO(str(annotations_file))

    detections = []

    with torch.no_grad():

        for images, targets in data_loader:

            images = [
                image.to(device)
                for image in images
            ]

            outputs = model(images)

            for output, target in zip(outputs, targets):

                image_id = int(
                    target["image_id"].item()
                )

                boxes = (
                    output["boxes"]
                    .detach()
                    .cpu()
                    .tolist()
                )

                scores = (
                    output["scores"]
                    .detach()
                    .cpu()
                    .tolist()
                )

                labels = (
                    output["labels"]
                    .detach()
                    .cpu()
                    .tolist()
                )

                for box, score, label in zip(
                    boxes,
                    scores,
                    labels,
                ):

                    # Ignore unexpected labels such as background.
                    if label not in LABEL_TO_COCO:
                        continue

                    x1, y1, x2, y2 = box

                    width = x2 - x1
                    height = y2 - y1

                    detections.append(
                        {
                            "image_id": image_id,
                            "category_id": LABEL_TO_COCO[label],
                            "bbox": [
                                x1,
                                y1,
                                width,
                                height,
                            ],
                            "score": float(score),
                        }
                    )

    if not detections:

        print(
            "No detections generated for evaluation."
        )

        return {}

    coco_dt = coco_gt.loadRes(detections)

    coco_eval = COCOeval(
        coco_gt,
        coco_dt,
        "bbox",
    )

    coco_eval.params.imgIds = sorted(
        coco_gt.getImgIds()
    )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats

    return {
        "mAP": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "mAR_1": float(stats[6]),
        "mAR_10": float(stats[7]),
        "mAR_100": float(stats[8]),
    }


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train the CarDD Faster R-CNN detector."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Training and validation batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of DataLoader workers.",
    )

    return parser.parse_args()


# ============================================================
# Training
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Checkpoints will be saved to: {MODELS_DIR}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Verify dataset
    # --------------------------------------------------------

    if not TRAIN_ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Training annotations not found: "
            f"{TRAIN_ANNOTATIONS_PATH}"
        )

    if not VAL_ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Validation annotations not found: "
            f"{VAL_ANNOTATIONS_PATH}"
        )

    if not TRAIN_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Training images not found: "
            f"{TRAIN_IMAGES_DIR}"
        )

    if not VAL_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Validation images not found: "
            f"{VAL_IMAGES_DIR}"
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = CarDDDetectionDataset(
        str(TRAIN_ANNOTATIONS_PATH),
        str(TRAIN_IMAGES_DIR),
    )

    val_dataset = CarDDDetectionDataset(
        str(VAL_ANNOTATIONS_PATH),
        str(VAL_IMAGES_DIR),
    )

    print(
        f"Training images: {len(train_dataset)}"
    )

    print(
        f"Validation images: {len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_detector()

    model = model.to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print()
        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        model.train()

        total_losses = []

        loss_components = {
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_objectness": [],
            "loss_rpn_box_reg": [],
        }

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for images, targets in train_loader:

            images = [
                image.to(device)
                for image in images
            ]

            targets = [
                {
                    key: value.to(device)
                    for key, value in target.items()
                }
                for target in targets
            ]

            # Faster R-CNN returns a dictionary
            # of training losses.
            loss_dict = model(
                images,
                targets,
            )

            loss = sum(
                loss_value
                for loss_value
                in loss_dict.values()
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_losses.append(
                loss.item()
            )

            for key in loss_components:

                if key in loss_dict:

                    loss_components[key].append(
                        loss_dict[key].item()
                    )

        # ----------------------------------------------------
        # Average epoch losses
        # ----------------------------------------------------

        avg_total = (
            sum(total_losses)
            / len(total_losses)
        )

        avg_classifier = (
            sum(
                loss_components[
                    "loss_classifier"
                ]
            )
            / len(
                loss_components[
                    "loss_classifier"
                ]
            )
        )

        avg_box_reg = (
            sum(
                loss_components[
                    "loss_box_reg"
                ]
            )
            / len(
                loss_components[
                    "loss_box_reg"
                ]
            )
        )

        avg_objectness = (
            sum(
                loss_components[
                    "loss_objectness"
                ]
            )
            / len(
                loss_components[
                    "loss_objectness"
                ]
            )
        )

        avg_rpn_box = (
            sum(
                loss_components[
                    "loss_rpn_box_reg"
                ]
            )
            / len(
                loss_components[
                    "loss_rpn_box_reg"
                ]
            )
        )

        print(
            f"total_loss={avg_total:.4f}"
        )

        print(
            f"classifier_loss="
            f"{avg_classifier:.4f}"
        )

        print(
            f"box_reg_loss="
            f"{avg_box_reg:.4f}"
        )

        print(
            f"objectness_loss="
            f"{avg_objectness:.4f}"
        )

        print(
            f"rpn_box_loss="
            f"{avg_rpn_box:.4f}"
        )

        # ----------------------------------------------------
        # Save BEFORE validation.
        #
        # Important:
        # Even if Colab disconnects during validation,
        # the trained epoch is already safe in Drive.
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            epoch,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        eval_metrics = evaluate_coco(
            model,
            val_loader,
            VAL_ANNOTATIONS_PATH,
            device,
        )

        if eval_metrics:

            print(
                f"val_mAP="
                f"{eval_metrics['mAP']:.4f}"
            )

            print(
                f"val_mAP_50="
                f"{eval_metrics['mAP_50']:.4f}"
            )

            print(
                f"val_mAP_75="
                f"{eval_metrics['mAP_75']:.4f}"
            )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()