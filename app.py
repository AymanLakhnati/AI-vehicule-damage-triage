import os
import socket
import sys
import hashlib
import json
import uuid
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from inference_service import analyze

FEEDBACK_DIR = Path(os.getenv("FEEDBACK_DIR", ROOT / "feedback_data"))

def classify(image):
    if image is None:
        return None, "### Add a vehicle image", "Upload a clear exterior photo to begin.", []
    try:
        result = analyze(image)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return image, "### Model unavailable", str(exc), []

    findings = result["findings"]
    assessment = result["assessment"]

    rows = [
        [item["name"].title(), f"{item['confidence']:.0%}", f"{item['threshold']:.0%}"]
        for item in findings
    ]
    if not rows:
        rows = [["No confident finding", "", ""]]

    finding_text = "\n".join(
        f"- **{item['name'].title()}** · {item['confidence']:.0%} confidence"
        for item in findings
    ) or "No damage class crossed its confidence threshold."
    summary = (
        f"**Model:** {result['model']} · **Device:** {result['device']}  \n"
        f"**Severity:** {assessment['severity']}  \n"
        f"**Next step:** {assessment['urgency']}  \n"
        f"**Indicative cost band:** {assessment['cost_band']}  \n\n"
        f"**Indicative Dubai range:** {assessment['indicative_cost_aed']}  \n\n"
        f"{assessment['guidance']}"
    )
    return result["image"], f"### {assessment['urgency'].title()}", summary + "\n\n" + finding_text, rows


def clear_image():
    return None, "### Add a vehicle image", "Upload a clear exterior photo to begin.", []


def submit_feedback(image, corrected_labels, consent):
    if image is None:
        return "Add an image before submitting feedback."
    if not consent:
        return "Nothing was saved. Enable consent if you want this example considered for training."
    if not corrected_labels or not corrected_labels.strip():
        return "Enter the corrected damage labels, or enter `none` if no damage is present."

    feedback_id = uuid.uuid4().hex
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    image_path = FEEDBACK_DIR / f"{feedback_id}.jpg"
    image.save(image_path, format="JPEG", quality=92)
    metadata = {
        "id": feedback_id,
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "corrected_labels": corrected_labels.strip(),
        "status": "pending_review",
    }
    (FEEDBACK_DIR / f"{feedback_id}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return "Feedback saved for human review. Thank you."


with gr.Blocks(title="AutoTriage") as demo:
    gr.HTML("""
    <section id="hero">
      <h1>AutoTriage</h1>
      <p>Upload one exterior vehicle photo for a cautious, explainable first-pass damage screen.</p>
    </section>
    """)
    with gr.Row():
        with gr.Column(scale=5, elem_classes="panel"):
            image_input = gr.Image(type="pil", label="Vehicle photo", height=390)
            with gr.Row():
                analyze = gr.Button("Analyze image", variant="primary", elem_id="analyze")
                clear = gr.Button("Clear", elem_id="clear")
            gr.Markdown("Use a well-lit image with the damaged area visible. Results are not a repair quote, insurance decision, or safety certification.")
            gr.Markdown("### Help improve the model")
            corrected_labels = gr.Textbox(
                label="Correct labels",
                placeholder="Example: dent, scratch",
                info="Only submit labels you have personally checked.",
            )
            consent = gr.Checkbox(
                label="I consent to storing this image and correction for model improvement.",
                value=False,
            )
            feedback_button = gr.Button("Submit reviewed correction")
            feedback_status = gr.Markdown()
        with gr.Column(scale=5, elem_classes="panel"):
            status = gr.Markdown("### Add a vehicle image\nUpload a clear exterior photo to begin.", elem_classes="status")
            guidance = gr.Markdown("Your result and cautious next step will appear here.")
            findings = gr.Dataframe(
                headers=["Damage type", "Confidence", "Threshold"],
                datatype=["str", "str", "str"],
                value=[],
                label="Findings",
                interactive=False,
            )

    analyze.click(classify, inputs=image_input, outputs=[image_input, status, guidance, findings])
    clear.click(clear_image, outputs=[image_input, status, guidance, findings])
    feedback_button.click(
        submit_feedback,
        inputs=[image_input, corrected_labels, consent],
        outputs=feedback_status,
    )

if __name__ == "__main__":
    requested_port = int(os.getenv("PORT", "7860"))
    with socket.socket() as probe:
        port_available = probe.connect_ex(("127.0.0.1", requested_port)) != 0
    if port_available:
        server_port = requested_port
    else:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            server_port = probe.getsockname()[1]
    print(f"Starting AutoTriage on http://127.0.0.1:{server_port}")
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=server_port,
        theme=gr.themes.Base(),
        css="""
:root { --ink: #17211b; --muted: #617067; --paper: #f4f0e8; --panel: #fffdf8; --accent: #2a9d8f; --warm: #e07a5f; }
body { background: #dce8df; }
.gradio-container { max-width: 1180px !important; margin: auto; padding: 28px 18px 42px !important; color: var(--ink); }
#hero { padding: 34px 36px 30px; border-radius: 18px; background: linear-gradient(135deg, #173b3f 0%, #2a6f68 58%, #8ab17d 100%); color: white; margin-bottom: 18px; box-shadow: 0 18px 42px rgba(23,59,63,.18); }
#hero h1 { font-size: 48px; letter-spacing: 0; margin: 0 0 8px; font-weight: 700; }
#hero p { margin: 0; max-width: 650px; color: #e3f0e7; font-size: 17px; }
.panel { background: var(--panel); border: 1px solid rgba(23,59,63,.12); border-radius: 14px; padding: 18px; box-shadow: 0 8px 22px rgba(23,59,63,.08); }
#analyze button { background: var(--warm); color: white; border: 0; font-weight: 700; }
#clear button { color: var(--ink); }
.status h3 { margin-bottom: 8px; }
footer { display: none !important; }
@media (max-width: 700px) { #hero h1 { font-size: 36px; } #hero { padding: 26px 22px; } }
        """,
    )
