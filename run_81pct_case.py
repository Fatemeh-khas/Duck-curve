"""One-command reproducible runner for the verified >=81% SSS case."""
from pathlib import Path

from run_paper_case import main


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(main([
        "--config", str(root / "configs" / "verified_81pct.json"),
        "--output-dir", str(root / "results_81pct"),
    ]))