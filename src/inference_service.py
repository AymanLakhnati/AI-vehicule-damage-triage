import json
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from cardd_detector import build_detector
from cardd_model import build_cardd_model
from triage_service import assess

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {
    1: "dent",
    2: "scratch",
    3: "crack",
    4: "glass shatter",
    5: "lamp broken",
    6: "tire flat",
}
CLASSIFIER_PATH = Path(os.getenv("CLASSIFIER_CHECKPOINT", ROOT / "models" / "cardd_resnet18_finetuned.pth"))
DETECTOR_PATH = Path(os.getenv("DETECTOR_CHECKPOINT", "")) if os.getenv("DETECTOR_CHECKPOINT") else None
CLASSIFIER_THRESHOLDS = ROOT / "models" / "cardd_thresholds.json"
DETECTOR_THRESHOLDS = ROOT / "models" / "cardd_detector_thresholds_epoch4.json"
_classifier = None
_detector = None


def _read_thresholds(path, fallback):
    if not path or not path.exists():
        return fallback
    return {**fallback, **json.loads(path.read_text(encoding="utf-8"))}


def _load_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    if not CLASSIFIER_PATH.exists():
        return None
    _classifier = build_cardd_model().to(DEVICE)
    _classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    _classifier.eval()
    return _classifier


def _load_detector():
    global _detector
    if _detector is not None:
        return _detector
    if not DETECTOR_PATH or not DETECTOR_PATH.exists():
        return None
    _detector = build_detector().to(DEVICE)
    _detector.load_state_dict(torch.load(DETECTOR_PATH, map_location=DEVICE))
    _detector.eval()
    return _detector


def _assessment(findings):
    decision = assess(findings)
    return {
        "severity": decision.severity,
        "urgency": decision.urgency,
        "guidance": decision.guidance,
        "cost_band": decision.cost_band,
        "requires_human_review": decision.requires_human_review,
    }


def _classifier_result(image):
    model = _load_classifier()
    if model is None:
        raise FileNotFoundError(f"Classifier checkpoint not found: {CLASSIFIER_PATH}")
    tensor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor))[0].cpu().tolist()
    fallback = {"dent": 0.40, "scratch": 0.30, "crack": 0.55, "glass shatter": 0.75, "lamp broken": 0.50, "tire flat": 0.80}
    thresholds = _read_thresholds(CLASSIFIER_THRESHOLDS, fallback)
    findings = [
        {"name": name, "confidence": round(score, 6), "threshold": thresholds[name]}
        for name, score in zip(CLASS_NAMES.values(), probabilities)
        if score >= thresholds[name]
    ]
    return findings, image.copy(), "classifier"


def _detector_result(image):
    model = _load_detector()
    if model is None:
        return None
    tensor = transforms.ToTensor()(image.convert("RGB")).to(DEVICE)
    with torch.no_grad():
        output = model([tensor])[0]
    fallback = {name: 0.50 for name in CLASS_NAMES.values()}
    thresholds = _read_thresholds(DETECTOR_THRESHOLDS, fallback)
    findings = []
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for box, label, score in zip(output["boxes"].cpu(), output["labels"].cpu(), output["scores"].cpu()):
        name = CLASS_NAMES.get(int(label))
        confidence = float(score)
        if name is None or confidence < thresholds[name]:
            continue
        findings.append({"name": name, "confidence": round(confidence, 6), "threshold": thresholds[name]})
        coordinates = [round(value, 1) for value in box.tolist()]
        draw.rectangle(coordinates, outline="#e07a5f", width=4)
        draw.text((coordinates[0], max(0, coordinates[1] - 18)), f"{name} {confidence:.0%}", fill="#e07a5f")
    return findings, annotated, "detector"


def analyze(image):
    if image is None:
        raise ValueError("An image is required.")
    result = _detector_result(image)
    if result is None:
        result = _classifier_result(image)
    findings, annotated, model_name = result
    findings.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "image": annotated,
        "findings": findings,
        "assessment": _assessment(findings),
        "model": model_name,
        "device": str(DEVICE),
    }
