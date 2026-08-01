import tempfile
import unittest
from pathlib import Path

from duckcurve.publication import publication_audit, source_checksums


class PublicationAuditTests(unittest.TestCase):
    def test_equal_budget_and_constraints_pass(self):
        row = {
            "evaluation_count": 100,
            "power_soc_cycle_limits_ok": True,
        }
        audit = publication_audit(
            seed_rows=[dict(row), dict(row)],
            selected_constraints={
                "soc_low": 0.0,
                "soc_high": 0.0,
                "soc_neutrality": 0.0,
                "power_limit": 0.0,
                "voltage": 0.0,
            },
            expected_seed_count=2,
            expected_evaluation_count=100,
            mcs_effects=[{"supports_improvement_at_95pct": False}],
        )
        self.assertTrue(audit["publication_checks_passed"])

    def test_unequal_budget_fails(self):
        rows = [
            {"evaluation_count": 100, "power_soc_cycle_limits_ok": True},
            {"evaluation_count": 101, "power_soc_cycle_limits_ok": True},
        ]
        audit = publication_audit(
            rows,
            {"voltage": 0.0},
            expected_seed_count=2,
            expected_evaluation_count=None,
            mcs_effects=[{"supports_improvement_at_95pct": True}],
        )
        self.assertFalse(audit["checks"]["equal_evaluation_budget"])
        self.assertFalse(audit["publication_checks_passed"])

    def test_source_manifest_excludes_temporary_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("x = 1\n", encoding="utf-8")
            (root / "tmp").mkdir()
            (root / "tmp" / "scratch.py").write_text("x = 2\n", encoding="utf-8")
            hashes = source_checksums(root)
            self.assertIn("model.py", hashes)
            self.assertNotIn("tmp/scratch.py", hashes)


if __name__ == "__main__":
    unittest.main()
