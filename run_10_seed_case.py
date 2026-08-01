"""One-command runner for the 10-seed SSS robustness study."""
from pathlib import Path

from run_paper_case import main


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(main([
        "--config", str(root / "configs" / "seed_validation_10.json"),
        "--output-dir", str(root / "results_10_seeds"),
    ]))