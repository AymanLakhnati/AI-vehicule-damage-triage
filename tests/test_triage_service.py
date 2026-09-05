import unittest

from src.triage_service import assess


class TriageServiceTests(unittest.TestCase):
    def test_safety_damage_is_urgent_and_requires_review(self):
        result = assess([{"name": "tire flat", "confidence": 0.94}])
        self.assertEqual(result.urgency, "urgent inspection")
        self.assertEqual(result.severity, "potentially safety-critical")
        self.assertTrue(result.requires_human_review)
        self.assertIn("low to medium", result.cost_band)
        self.assertEqual(result.indicative_cost_aed, "AED 250-1,500")

    def test_uncertain_finding_requires_review(self):
        result = assess([{"name": "crack", "confidence": 0.51}])
        self.assertEqual(result.urgency, "inspect soon")
        self.assertTrue(result.requires_human_review)

    def test_empty_findings_are_cautious(self):
        result = assess([])
        self.assertEqual(result.cost_band, "unknown")
        self.assertEqual(result.urgency, "monitor and inspect")


if __name__ == "__main__":
    unittest.main()
