import io
import tempfile
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from api import app
import api


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_contract(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("X-Request-ID", response.headers)
        self.assertIn("X-Response-Time-Ms", response.headers)

    def test_ready_contract(self):
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")

    def test_metrics_requires_admin_authentication(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 401)

    def test_request_id_is_preserved(self):
        response = self.client.get("/health", headers={"X-Request-ID": "request-test-1"})
        self.assertEqual(response.headers["X-Request-ID"], "request-test-1")

    def test_rejects_non_image_upload(self):
        response = self.client.post(
            "/v1/analyze",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    def test_feedback_without_consent_does_not_retain_image(self):
        response = self.client.post(
            "/v1/feedback",
            files={"file": ("vehicle.jpg", b"image bytes", "image/jpeg")},
            data={"consent_to_training": "false", "corrected_labels": "dent"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["accepted"])

    def test_feedback_rejects_unknown_labels(self):
        image = Image.new("RGB", (8, 8), "white")
        payload = io.BytesIO()
        image.save(payload, format="JPEG")
        response = self.client.post(
            "/v1/feedback",
            files={"file": ("vehicle.jpg", payload.getvalue(), "image/jpeg")},
            data={"consent_to_training": "true", "corrected_labels": "engine failure"},
        )
        self.assertEqual(response.status_code, 400)

    def test_consented_feedback_image_is_retrievable_by_admin(self):
        image = Image.new("RGB", (16, 16), "white")
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        with tempfile.TemporaryDirectory() as directory:
            original_dir = api.FEEDBACK_DIR
            original_token = api.ADMIN_TOKEN
            original_store = api.STORE
            api.FEEDBACK_DIR = __import__("pathlib").Path(directory)
            api.ADMIN_TOKEN = "test-admin-token"
            api.STORE = __import__("src.storage", fromlist=["Storage"]).Storage(__import__("pathlib").Path(directory) / "autotriage.db")
            try:
                response = self.client.post(
                    "/v1/feedback",
                    files={"file": ("vehicle.png", payload.getvalue(), "image/png")},
                    data={"consent_to_training": "true", "corrected_labels": "dent"},
                )
                self.assertEqual(response.status_code, 200)
                feedback_id = response.json()["feedback_id"]
                image_response = self.client.get(
                    f"/v1/feedback/{feedback_id}/image",
                    headers={"X-Admin-Token": "test-admin-token"},
                )
                self.assertEqual(image_response.status_code, 200)
                self.assertEqual(image_response.headers["content-type"], "image/jpeg")
            finally:
                api.FEEDBACK_DIR = original_dir
                api.ADMIN_TOKEN = original_token
                api.STORE = original_store

    def test_feedback_queue_requires_admin_authentication(self):
        response = self.client.get("/v1/feedback/pending")
        self.assertEqual(response.status_code, 401)

    def test_partners_never_returns_unconfigured_listings(self):
        response = self.client.get("/v1/partners", params={"city": "Dubai"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["partners"], [])

    def test_booking_requires_location_consent(self):
        response = self.client.post(
            "/v1/booking-requests",
            data={
                "partner_id": "missing",
                "contact_name": "Test User",
                "contact_phone": "+971500000000",
                "preferred_time": "Tomorrow morning",
                "location_consent": "false",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_booking_rejects_unverified_partner(self):
        response = self.client.post(
            "/v1/booking-requests",
            data={
                "partner_id": "missing",
                "contact_name": "Test User",
                "contact_phone": "+971500000000",
                "preferred_time": "Tomorrow morning",
                "location_consent": "true",
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_booking_creates_request_for_verified_partner(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = __import__("pathlib").Path(directory)
            partners_path = directory_path / "partners.json"
            partners_path.write_text(
                '[{"id": "dubai-1", "name": "Verified Shop", "city": "Dubai", "verified": true}]',
                encoding="utf-8",
            )
            original_partners = api.PARTNERS_PATH
            original_store = api.STORE
            api.PARTNERS_PATH = partners_path
            api.STORE = __import__("src.storage", fromlist=["Storage"]).Storage(directory_path / "autotriage.db")
            try:
                response = self.client.post(
                    "/v1/booking-requests",
                    data={
                        "partner_id": "dubai-1",
                        "contact_name": "Test User",
                        "contact_phone": "+971500000000",
                        "preferred_time": "Tomorrow morning",
                        "location_consent": "true",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "requested")
            finally:
                api.PARTNERS_PATH = original_partners
                api.STORE = original_store

    def test_admin_dashboard_is_available_without_exposing_feedback(self):
        response = self.client.get("/admin")
        self.assertEqual(response.status_code, 200)
        self.assertIn("AutoTriage review queue", response.text)

    def test_export_requires_admin_authentication(self):
        response = self.client.get("/v1/feedback/export")
        self.assertEqual(response.status_code, 401)

    def test_analyze_accepts_image(self):
        image = Image.new("RGB", (32, 32), "white")
        payload = io.BytesIO()
        image.save(payload, format="JPEG")
        response = self.client.post(
            "/v1/analyze",
            files={"file": ("vehicle.jpg", payload.getvalue(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("findings", body)
        self.assertIn("assessment", body)
        self.assertIn("assessment_id", body)
        saved = self.client.get(f"/v1/assessments/{body['assessment_id']}")
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["id"], body["assessment_id"])

    def test_unknown_assessment_returns_not_found(self):
        response = self.client.get("/v1/assessments/does-not-exist")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
