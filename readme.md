# AutoTriage

Private vehicle-damage assessment prototype for cautious image-based screening.

## Local web app

```powershell
python -m pip install -r requirements.txt
python app.py
```

The app uses the local classification checkpoint by default. To use the stronger detector, configure its private checkpoint before launch:

```powershell
$env:DETECTOR_CHECKPOINT="C:\path\to\cardd_detector_epoch4.pth"
python app.py
```

## API

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `GET /ready`
- `GET /metrics` with `X-Admin-Token`
- `POST /v1/analyze`
- `GET /v1/assessments/{assessment_id}`
- `POST /v1/feedback` with explicit training consent
- `GET /v1/partners?city=Dubai`
- `POST /v1/booking-requests` with location consent and a verified partner ID
- `GET /v1/booking-requests` with `X-Admin-Token`

## Docker deployment

Model weights are intentionally excluded from the image. Mount a private model directory at runtime:

```powershell
docker build -t autotriage-api .
docker run --rm -p 8000:8000 `
	-e ADMIN_TOKEN="use-a-long-secret" `
	-e DETECTOR_CHECKPOINT="/models/cardd_detector_epoch4.pth" `
	-v "C:\private\autotriage-models:/models:ro" `
	autotriage-api
```

For classifier-only mode, mount `cardd_resnet18_finetuned.pth` and set `CLASSIFIER_CHECKPOINT=/models/cardd_resnet18_finetuned.pth`.

## First cloud deployment

`render.yaml` provisions the API with a persistent disk and `/ready` health checks. The private disk must contain the model files at:

```text
/data/models/cardd_resnet18_finetuned.pth
```

The first Render release runs with the classifier fallback. After uploading `cardd_detector_epoch4.pth` to `/data/models`, add `DETECTOR_CHECKPOINT=/data/models/cardd_detector_epoch4.pth` in Render and redeploy.

Set `ADMIN_TOKEN`, `APP_API_KEY`, and `CORS_ORIGINS` as secret environment variables in the hosting dashboard. Do not commit model weights or secrets.

## Curated feedback export

After reviewing submissions in `/admin`, export only approved labels:

```powershell
python src/export_approved_feedback.py
```

This creates `data/approved_feedback/labels.csv` and copies the approved images into its `images/` directory. Unreviewed and rejected submissions are excluded.

Fine-tune the classifier with the curated set:

```powershell
python src/train_cardd_with_feedback.py --epochs 5
```

The best checkpoint is selected using `val2017`; the untouched test split is not used during this training run.
Each run writes `reports/feedback_training_run.json` with the base-model hash, approved-example count, hyperparameters, best validation loss, and a `test_split_used: false` record.

## Mobile client

The Expo client is in `mobile/`. Set `EXPO_PUBLIC_API_URL` in `mobile/.env`, then run:

```powershell
cd mobile
npm install
npm start
```

For device builds, install EAS CLI and run `eas build --platform all --profile production`. Set `EXPO_PUBLIC_API_URL` to the deployed HTTPS API URL before building.

## Data boundary

The untouched test split is used only for final evaluation. Feedback images are retained only when the user explicitly consents to training. Cost output is an indicative category, not a quotation. Partner listings must be verified before being configured.

Feedback and booking metadata use SQLite for this private prototype through `DATABASE_PATH`. Before horizontal scaling, migrate the storage adapter to managed PostgreSQL and move images to private object storage.
