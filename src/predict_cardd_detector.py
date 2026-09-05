import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

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

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = Path(
    "/content/drive/MyDrive/vehicle-damage-triage-models/"
    "cardd_detector_epoch3.pth"
)
OUTPUT_DIR = REPO_ROOT / "out"
CONFIDENCE_THRESHOLD = 0.50


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the detector model from checkpoint."""
    model = build_detector().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_and_preprocess_image(image_path: Path, device: torch.device) -> tuple:
    """Load image and convert to tensor."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(image).to(device)
    return image, image_tensor


def predict_single_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    confidence_threshold: float = 0.50,
) -> dict:
    """Run inference on a single image."""
    image, image_tensor = load_and_preprocess_image(image_path, device)

    with torch.no_grad():
        outputs = model([image_tensor])

    # Extract predictions from output
    predictions = {
        "boxes": outputs[0]["boxes"].cpu().numpy(),
        "labels": outputs[0]["labels"].cpu().numpy(),
        "scores": outputs[0]["scores"].cpu().numpy(),
    }

    # Filter by confidence threshold
    mask = predictions["scores"] >= confidence_threshold
    predictions["boxes"] = predictions["boxes"][mask]
    predictions["labels"] = predictions["labels"][mask]
    predictions["scores"] = predictions["scores"][mask]

    return image, predictions


def draw_predictions(
    image: Image.Image,
    predictions: dict,
    class_names: dict,
    output_path: Path,
) -> None:
    """Draw bounding boxes and labels on image."""
    draw = ImageDraw.Draw(image)

    # Try to load a default font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    # Define colors for bounding boxes (RGB)
    colors = {
        1: (255, 0, 0),  # Red for dent
        2: (0, 255, 0),  # Green for scratch
        3: (0, 0, 255),  # Blue for crack
        4: (255, 255, 0),  # Yellow for glass shatter
        5: (255, 0, 255),  # Magenta for lamp broken
        6: (0, 255, 255),  # Cyan for tire flat
    }

    boxes = predictions["boxes"]
    labels = predictions["labels"]
    scores = predictions["scores"]

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        label_name = class_names.get(int(label), f"class_{label}")
        color = colors.get(int(label), (255, 255, 255))

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label with score
        label_text = f"{label_name}: {score:.2f}"
        # Get text bounding box to create a background
        bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), label_text, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Predict vehicle damage on a single image using Epoch 3 detector."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input car image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="out/detection_prediction.jpg",
        help="Path to save the annotated output image (default: out/detection_prediction.jpg).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Confidence threshold for detections (default: 0.50).",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = load_model(CHECKPOINT_PATH, device)

    # Load image and run prediction
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Processing image: {image_path}")
    image, predictions = predict_single_image(
        image_path, model, device, confidence_threshold=args.confidence
    )

    # Print results
    print("\nDetected damages:")
    class_scores = {}
    for label, score in zip(predictions["labels"], predictions["scores"]):
        label_name = CLASS_NAMES.get(int(label), f"class_{label}")
        if label_name not in class_scores:
            class_scores[label_name] = []
        class_scores[label_name].append(score)

    for class_name in sorted(class_scores.keys()):
        avg_score = sum(class_scores[class_name]) / len(class_scores[class_name])
        count = len(class_scores[class_name])
        print(f"  - {class_name}: {avg_score:.2f} (detected {count} times)")

    if not predictions["labels"].size:
        print("  (No damage detected above confidence threshold)")

    # Draw predictions and save
    output_path = Path(args.output)
    draw_predictions(image, predictions, CLASS_NAMES, output_path)
    print(f"\nSaved prediction to: {output_path}")


if __name__ == "__main__":
    main()
