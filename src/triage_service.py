from dataclasses import dataclass
from typing import Any


SAFETY_CLASSES = {"glass shatter", "lamp broken", "tire flat"}
COST_BANDS = {
    "dent": ("medium", "AED 500-2,500"),
    "scratch": ("low to medium", "AED 300-2,000"),
    "crack": ("medium to high", "AED 800-4,000"),
    "glass shatter": ("high", "AED 1,500-8,000"),
    "lamp broken": ("medium to high", "AED 700-4,000"),
    "tire flat": ("low to medium", "AED 250-1,500"),
}


@dataclass(frozen=True)
class Assessment:
    severity: str
    urgency: str
    guidance: str
    cost_band: str
    indicative_cost_aed: str
    requires_human_review: bool


def assess(findings: list[dict[str, Any]]) -> Assessment:
    names = {item["name"] for item in findings}
    uncertain = any(item["confidence"] < 0.70 for item in findings)
    requires_review = uncertain or bool(names & SAFETY_CLASSES)

    if names & SAFETY_CLASSES:
        severity = "potentially safety-critical"
        urgency = "urgent inspection"
        guidance = "Do not rely on this image assessment to decide that the vehicle is safe to drive. Arrange professional inspection, especially if visibility, lighting, or tire control may be affected."
    elif any(item["name"] in {"crack", "dent"} for item in findings):
        severity = "moderate visible damage"
        urgency = "inspect soon"
        guidance = "Arrange an inspection to confirm structural or panel damage and determine the repair scope."
    elif findings:
        severity = "minor visible damage"
        urgency = "routine inspection"
        guidance = "A professional should confirm the affected area and repair requirements."
    else:
        severity = "no confident finding"
        urgency = "monitor and inspect"
        guidance = "No damage class passed the confidence threshold. A clean result does not guarantee that damage is absent."

    bands = sorted({COST_BANDS[item["name"]][0] for item in findings})
    ranges = sorted({COST_BANDS[item["name"]][1] for item in findings})
    cost_band = "; ".join(bands) if bands else "unknown"
    indicative_cost_aed = "; ".join(ranges) if ranges else "unknown"
    return Assessment(severity, urgency, guidance, cost_band, indicative_cost_aed, requires_review)
