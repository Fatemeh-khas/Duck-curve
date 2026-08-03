"""Reproducibility and release-audit utilities for publication runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_checksums(root: Path) -> dict[str, str]:
    files = []
    for pattern in ("*.py", "*.yaml", "*.json", "*.md", "*.txt"):
        files.extend(root.rglob(pattern))
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(set(files))
        if "__pycache__" not in path.parts
        and not any(
            part == "tmp"
            or part == "previous results singles seeds"
            or part.startswith(("outputs", "results"))
            for part in path.parts
        )
    }


def dependency_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(root: Path, config_path: Path, config: dict) -> dict:
    return {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "completed_utc": None,
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": git_revision(root),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "config": config,
        "dependencies": dependency_versions(
            ["numpy", "matplotlib", "PyYAML", "pandapower"]
        ),
        "source_sha256": source_checksums(root),
    }


def finalize_manifest(
    manifest: dict,
    output_dir: Path,
    audit: dict,
) -> dict:
    manifest = dict(manifest)
    manifest["status"] = "complete" if audit["publication_checks_passed"] else "failed_audit"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["audit"] = audit
    manifest["output_sha256"] = {
        str(path.relative_to(output_dir)).replace("\\", "/"): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    return manifest


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publication_audit(
    seed_rows: list[dict],
    selected_constraints: dict,
    expected_seed_count: int,
    expected_evaluation_count: int | None,
    mcs_effects: list[dict] | None,
    require_supported_reliability_improvement: bool = False,
    cvar_constraint_satisfied: bool = True,
    tolerance: float = 1.0e-7,
) -> dict:
    evaluation_counts = [int(row["evaluation_count"]) for row in seed_rows]
    checks = {
        "expected_seed_count_completed": len(seed_rows) == expected_seed_count,
        "all_seed_constraints_pass": all(
            bool(row["power_soc_cycle_limits_ok"]) for row in seed_rows
        ),
        "equal_evaluation_budget": len(set(evaluation_counts)) == 1,
        "declared_evaluation_budget_matches": (
            expected_evaluation_count is None
            or all(count == expected_evaluation_count for count in evaluation_counts)
        ),
        "selected_soc_low_ok": float(selected_constraints.get("soc_low", 0.0)) <= tolerance,
        "selected_soc_high_ok": float(selected_constraints.get("soc_high", 0.0)) <= tolerance,
        "selected_soc_neutrality_ok": (
            float(selected_constraints.get("soc_neutrality", 0.0)) <= tolerance
        ),
        "selected_power_limit_ok": (
            float(selected_constraints.get("power_limit", 0.0)) <= tolerance
        ),
        "selected_voltage_ok": float(selected_constraints.get("voltage", 0.0)) <= tolerance,
        "reliability_outputs_present": mcs_effects is not None,
        "cvar_constraint_satisfied": bool(cvar_constraint_satisfied),
    }
    if require_supported_reliability_improvement:
        checks["reliability_improvement_supported"] = bool(
            mcs_effects and mcs_effects[-1]["supports_improvement_at_95pct"]
        )
    return {
        "publication_checks_passed": all(checks.values()),
        "checks": checks,
        "evaluation_counts_by_seed": evaluation_counts,
        "note": (
            "A failed reliability-improvement hypothesis is not a failed run unless "
            "the preregistered configuration explicitly requires that hypothesis."
        ),
    }
