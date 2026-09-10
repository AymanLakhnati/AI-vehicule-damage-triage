import hashlib
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from src.inference_service import analyze as run_inference
from src.storage import Storage

FEEDBACK_DIR = Path(os.getenv("FEEDBACK_DIR", ROOT / "feedback_data"))
PARTNERS_PATH = Path(os.getenv("PARTNERS_PATH", ROOT / "config" / "verified_partners.json"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", ROOT / "data" / "autotriage.db"))
STORE = Storage(DATABASE_PATH)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
APP_API_KEY = os.getenv("APP_API_KEY")
DEVICE = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
REQUEST_METRICS = {"total": 0, "errors": 0, "analysis_total": 0, "analysis_errors": 0, "analysis_seconds": 0.0}

app = FastAPI(title="AutoTriage API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8081").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request, call_next):
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    started = time.perf_counter()
    REQUEST_METRICS["total"] += 1
    try:
        response = await call_next(request)
    except Exception:
        REQUEST_METRICS["errors"] += 1
        raise
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def decode_image(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload))
        image.verify()
        image = Image.open(io.BytesIO(payload))
        if image.width * image.height > 25_000_000:
            raise HTTPException(status_code=413, detail="Image dimensions are too large.")
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc


def validate_labels(value: str) -> str:
    labels = [label.strip().lower() for label in value.split(",") if label.strip()]
    allowed = {"dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat", "none"}
    if not labels or any(label not in allowed for label in labels):
        raise HTTPException(status_code=400, detail="Corrected labels must use the supported damage names or none.")
    if "none" in labels and len(labels) > 1:
        raise HTTPException(status_code=400, detail="Use none by itself when no damage is present.")
    return ", ".join(dict.fromkeys(labels))


def require_app_key(api_key: str | None):
    if APP_API_KEY and api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Application authentication required.")


@app.get("/")
def root():
    return {
        "service": "AutoTriage API",
        "status": "online",
        "health": "/health",
        "readiness": "/ready",
        "documentation": "/docs",
        "analysis": "/v1/analyze",
    }


@app.get("/health")
def health():
    classifier_path = Path(os.getenv("CLASSIFIER_CHECKPOINT", ROOT / "models" / "cardd_resnet18_finetuned.pth"))
    detector_path = os.getenv("DETECTOR_CHECKPOINT")
    return {
        "status": "ok",
        "classifier_available": classifier_path.exists(),
        "detector_available": bool(detector_path and Path(detector_path).exists()),
        "device": str(DEVICE),
    }


@app.get("/ready")
def ready():
    classifier_path = Path(os.getenv("CLASSIFIER_CHECKPOINT", ROOT / "models" / "cardd_resnet18_finetuned.pth"))
    detector_path = os.getenv("DETECTOR_CHECKPOINT")
    if not classifier_path.exists() and not (detector_path and Path(detector_path).exists()):
        raise HTTPException(status_code=503, detail="No inference model is configured.")
    return {"status": "ready", "device": str(DEVICE)}


@app.get("/metrics")
def metrics(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    return {**REQUEST_METRICS}


@app.post("/v1/analyze")
async def analyze(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    require_app_key(x_api_key)
    REQUEST_METRICS["analysis_total"] += 1
    analysis_started = time.perf_counter()
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    payload = await file.read()
    if len(payload) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 12 MB or smaller.")
    image = decode_image(payload)
    try:
        result = run_inference(image)
    except FileNotFoundError as exc:
        REQUEST_METRICS["analysis_errors"] += 1
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        REQUEST_METRICS["analysis_errors"] += 1
        raise
    REQUEST_METRICS["analysis_seconds"] += time.perf_counter() - analysis_started
    result.pop("image", None)
    assessment_id = uuid.uuid4().hex
    result["assessment_id"] = assessment_id
    STORE.add_assessment(assessment_id, result)
    return result


@app.get("/v1/assessments/{assessment_id}")
def get_assessment(assessment_id: str, x_api_key: str | None = Header(default=None)):
    require_app_key(x_api_key)
    assessment = STORE.get_assessment(assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return assessment


@app.post("/v1/feedback")
async def feedback(
    file: UploadFile = File(...),
    consent_to_training: bool = Form(False),
    corrected_labels: str = Form(...),
    x_api_key: str | None = Header(default=None),
):
    require_app_key(x_api_key)
    if not consent_to_training:
        return {"accepted": False, "message": "No image was retained because training consent was not provided."}
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    payload = await file.read()
    if len(payload) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 12 MB or smaller.")
    image = decode_image(payload)
    corrected_labels = validate_labels(corrected_labels)
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    feedback_id = uuid.uuid4().hex
    image_path = FEEDBACK_DIR / f"{feedback_id}.jpg"
    image.save(image_path, format="JPEG", quality=92)
    stored_payload = image_path.read_bytes()
    STORE.add_feedback({
            "id": feedback_id,
            "sha256": hashlib.sha256(stored_payload).hexdigest(),
            "corrected_labels": corrected_labels,
            "status": "pending_review",
    })
    return {"accepted": True, "feedback_id": feedback_id, "status": "pending_review"}


def require_admin(token: str | None):
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Administrator authentication required.")


@app.get("/v1/feedback/pending")
def pending_feedback(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    return {"items": STORE.list_feedback("pending_review")}


@app.post("/v1/feedback/{feedback_id}/review")
def review_feedback(feedback_id: str, status: str, approved_labels: str | None = None, x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    if status not in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="Status must be approved or rejected.")
    records = [item for item in STORE.list_feedback() if item["id"] == feedback_id]
    if not records:
        raise HTTPException(status_code=404, detail="Feedback item not found.")
    STORE.review_feedback(feedback_id, status, approved_labels.strip() if approved_labels is not None else None)
    return {"feedback_id": feedback_id, "status": status}


@app.get("/v1/feedback/{feedback_id}/image")
def feedback_image(feedback_id: str, x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    image_path = FEEDBACK_DIR / f"{feedback_id}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Feedback image not found.")
    return Response(content=image_path.read_bytes(), media_type="image/jpeg")


@app.get("/v1/feedback/export")
def export_feedback(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    approved = [record for record in STORE.list_feedback("approved") if record.get("approved_labels")]
    return {"items": approved, "count": len(approved)}


@app.get("/admin")
def admin_dashboard():
    return Response(
        content="""<!doctype html>
<html><head><meta charset="utf-8"><title>AutoTriage Review</title>
<style>body{font:16px system-ui;max-width:960px;margin:40px auto;background:#edf2e9;color:#173b3f}button,input{padding:10px;margin:4px;border:1px solid #9bb5a6;border-radius:6px}button{background:#e07a5f;color:#fff;border:0}.item{background:#fffdf8;padding:16px;margin:12px 0;border-radius:10px}.item img{max-width:260px;max-height:180px;display:block;margin:10px 0}.muted{color:#617067}</style></head>
<body><h1>AutoTriage review queue</h1><p class="muted">Private workspace. Approve only labels you have personally verified.</p>
<input id="token" type="password" placeholder="Admin token"><button onclick="loadQueue()">Load pending</button><button onclick="exportData()">Export approved</button><div id="status"></div><main id="queue"></main>
<script>
const api = location.origin;
function headers(){return {'X-Admin-Token':document.getElementById('token').value};}
async function loadQueue(){const r=await fetch(api+'/v1/feedback/pending',{headers:headers()});if(!r.ok){document.getElementById('status').textContent='Authentication failed.';return;}const data=await r.json();const queue=document.getElementById('queue');queue.replaceChildren();data.items.forEach(item=>{const el=document.createElement('section');el.className='item';const title=document.createElement('strong');title.textContent=item.id;const submitted=document.createElement('p');submitted.textContent='Submitted labels: '+item.corrected_labels;const image=document.createElement('img');const label=document.createElement('input');label.value=item.corrected_labels;const approve=document.createElement('button');approve.textContent='Approve';const reject=document.createElement('button');reject.textContent='Reject';el.append(title,submitted,image,label,approve,reject);fetch(api+'/v1/feedback/'+encodeURIComponent(item.id)+'/image',{headers:headers()}).then(response=>response.blob()).then(blob=>{image.src=URL.createObjectURL(blob);});approve.onclick=()=>review(item.id,'approved',label.value);reject.onclick=()=>review(item.id,'rejected',label.value);queue.appendChild(el);});document.getElementById('status').textContent=data.items.length+' pending item(s).';}
async function review(id,status,labels){const r=await fetch(api+'/v1/feedback/'+id+'/review?status='+status+'&approved_labels='+encodeURIComponent(labels),{method:'POST',headers:headers()});if(r.ok)loadQueue();}
async function exportData(){const r=await fetch(api+'/v1/feedback/export',{headers:headers()});const data=await r.json();document.getElementById('status').textContent=data.count+' approved item(s) ready for retraining.';console.log(data);}
</script></body></html>""",
        media_type="text/html",
    )


@app.get("/v1/partners")
def partners(city: str = "Dubai"):
    if not PARTNERS_PATH.exists():
        return {"city": city, "partners": [], "message": "No verified partners are configured yet."}
    configured = json.loads(PARTNERS_PATH.read_text(encoding="utf-8"))
    matches = [partner for partner in configured if partner.get("city", "").lower() == city.lower() and partner.get("verified") is True]
    return {"city": city, "partners": matches, "message": "Listings are provided only for verified partners."}


@app.post("/v1/booking-requests")
def create_booking_request(
    partner_id: str = Form(...),
    contact_name: str = Form(...),
    contact_phone: str = Form(...),
    preferred_time: str = Form(...),
    city: str = Form("Dubai"),
    location_consent: bool = Form(False),
    assessment_id: str | None = Form(default=None),
    x_api_key: str | None = Header(default=None),
):
    require_app_key(x_api_key)
    if not location_consent:
        raise HTTPException(status_code=400, detail="Location consent is required to request a visit.")
    if not contact_name.strip() or not contact_phone.strip() or not preferred_time.strip():
        raise HTTPException(status_code=400, detail="Name, phone, and preferred time are required.")
    if not PARTNERS_PATH.exists():
        raise HTTPException(status_code=404, detail="No verified partners are available.")
    configured = json.loads(PARTNERS_PATH.read_text(encoding="utf-8"))
    partner = next(
        (
            item for item in configured
            if str(item.get("id")) == partner_id
            and item.get("city", "").lower() == city.lower()
            and item.get("verified") is True
        ),
        None,
    )
    if partner is None:
        raise HTTPException(status_code=404, detail="Verified partner not found for this city.")

    request_id = uuid.uuid4().hex
    record = {
        "id": request_id,
        "partner_id": partner_id,
        "city": city,
        "contact_name": contact_name.strip(),
        "contact_phone": contact_phone.strip(),
        "preferred_time": preferred_time.strip(),
        "assessment_id": assessment_id,
        "location_consent": True,
        "status": "requested",
    }
    STORE.add_booking(record)
    return record


@app.get("/v1/booking-requests")
def booking_requests(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    return {"items": STORE.list_bookings()}
