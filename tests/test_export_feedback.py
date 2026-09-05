import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.export_approved_feedback import export_feedback


class ApprovedFeedbackExportTests(unittest.TestCase):
    def test_exports_only_approved_records(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback = Path(directory) / "feedback"
            output = Path(directory) / "output"
            feedback.mkdir()
            Image.new("RGB", (8, 8), "white").save(feedback / "approved-id.jpg")
            Image.new("RGB", (8, 8), "black").save(feedback / "pending-id.jpg")
            (feedback / "approved-id.json").write_text(json.dumps({
                "id": "approved-id",
                "status": "approved",
                "approved_labels": "crack, dent",
            }), encoding="utf-8")
            (feedback / "pending-id.json").write_text(json.dumps({
                "id": "pending-id",
                "status": "pending_review",
                "corrected_labels": "scratch",
            }), encoding="utf-8")

            manifest, count, skipped = export_feedback(feedback, output)

            self.assertEqual(count, 1)
            self.assertEqual(skipped, [])
            self.assertTrue(manifest.exists())
            text = manifest.read_text(encoding="utf-8")
            self.assertIn("approved-id.jpg", text)
            self.assertIn(",1,", text)
            self.assertTrue((output / "images" / "approved-id.jpg").exists())
            self.assertFalse((output / "images" / "pending-id.jpg").exists())


if __name__ == "__main__":
    unittest.main()
