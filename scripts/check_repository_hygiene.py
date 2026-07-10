"""Validate the tracked-file contract for the GestureBind MVP release."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_TRACKED_PREFIXES = (
    ".db_data/",
    ".idea/",
    ".pytest_cache/",
    ".venv/",
    ".vscode/",
    "app/qml/",
    "data/",
    "htmlcov/",
    "mlruns/",
    "models/experiments/",
    "models/user_gestures/",
    "node_modules/",
    "outputs/",
    "tmp/",
    "venv/",
)

FORBIDDEN_TRACKED_PATHS = {
    ".coverage",
    ".env",
    "app/gui_main.py",
    "app/main.py",
    "app/n.py",
    "mlflow.db",
    "package.json",
    "scripts/launch_app_fixed.sh",
}

FORBIDDEN_BASENAMES = {".DS_Store"}
FORBIDDEN_SUFFIXES = (".doc", ".docx", ".pdf", ".ppt", ".pptx", ".pyc")

RELEASE_MODEL_FILES = {
    "models/classes.json",
    "models/dynamic_landmark_lstm_backbone.pkl",
    "models/dynamic_landmark_lstm_backbone_classes.json",
    "models/dynamic_landmark_lstm_backbone_feature_dim.txt",
    "models/dynamic_landmark_lstm_backbone_feature_mode.txt",
    "models/dynamic_landmark_lstm_backbone_optuna.json",
    "models/dynamic_landmark_lstm_backbone_prototypes.json",
    "models/dynamic_landmark_lstm_backbone_rejection.json",
    "models/feature_dim.txt",
    "models/feature_mode.txt",
    "models/gesture_rejection.json",
    "models/intent_gate_metadata.json",
    "models/intent_gate_mlp.pkl",
    "models/knn.pkl",
    "models/knn_optuna.json",
}

REQUIRED_RELEASE_FILES = {
    ".env.example",
    ".github/workflows/ci.yml",
    ".github/workflows/desktop-release.yml",
    ".gitignore",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
    "LICENSE",
    "Makefile",
    "packaging/macos/RUN_MACOS.md",
    "packaging/macos/build_app.sh",
    "packaging/macos/gesturebind_launcher.py",
    "README.md",
    "RELEASE_NOTES.md",
    "VERSION",
    "app/flet_app/main.py",
    "configs/gesture_taxonomy.json",
    "docs/contest/ML_SYSTEM_DESIGN.md",
    "docs/experiments/release_readiness_2026-07-10.md",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-research.txt",
    "scripts/check_repository_hygiene.py",
    "scripts/launch_app.sh",
    "scripts/ml_smoke.py",
    *RELEASE_MODEL_FILES,
}

ALLOWED_EXPERIMENT_EVIDENCE = {
    "docs/experiments/README.md",
    "docs/experiments/dynamic_lstm_grouped_retrain_2026-07-10.json",
    "docs/experiments/dynamic_lstm_grouped_retrain_2026-07-10.md",
    "docs/experiments/dynamic_completion_benchmark.json",
    "docs/experiments/dynamic_completion_benchmark.md",
    "docs/experiments/ml_pipeline_log.md",
    "docs/experiments/release_latency_benchmark_2026-07-10.json",
    "docs/experiments/release_readiness_2026-07-10.md",
    "docs/experiments/static_grouped_cv_release_2026-07-10.json",
    "docs/experiments/static_grouped_cv_release_2026-07-10.md",
}


def tracked_files(root: Path = PROJECT_ROOT) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    }


def validate_tracked_files(paths: Iterable[str]) -> list[str]:
    normalized: set[str] = set()
    for path in paths:
        value = str(path).replace("\\", "/")
        normalized.add(value[2:] if value.startswith("./") else value)
    errors: list[str] = []

    for path in sorted(normalized):
        if path in FORBIDDEN_TRACKED_PATHS:
            errors.append(f"forbidden tracked file: {path}")
        if Path(path).name in FORBIDDEN_BASENAMES:
            errors.append(f"generated metadata is tracked: {path}")
        if path.lower().endswith(FORBIDDEN_SUFFIXES):
            errors.append(f"generated/binary document is tracked: {path}")
        for prefix in FORBIDDEN_TRACKED_PREFIXES:
            if path.startswith(prefix):
                errors.append(f"forbidden tracked directory: {path}")
                break

        if path.startswith("models/") and path not in RELEASE_MODEL_FILES:
            errors.append(f"unexpected release model artifact: {path}")
        if (
            path.startswith("docs/experiments/")
            and path not in ALLOWED_EXPERIMENT_EVIDENCE
        ):
            errors.append(f"generated experiment evidence is tracked: {path}")

    for required in sorted(REQUIRED_RELEASE_FILES - normalized):
        errors.append(f"required release file is missing: {required}")

    return errors


def validate_release_metadata(root: Path = PROJECT_ROOT) -> list[str]:
    errors: list[str] = []
    try:
        version = (root / "VERSION").read_text(encoding="utf-8").strip()
    except OSError as exc:
        return [f"unable to read VERSION: {exc}"]

    if re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        errors.append(f"VERSION is not semantic x.y.z: {version!r}")
        return errors

    contracts = {
        "README.md": f"MVP `{version}`",
        "RELEASE_NOTES.md": f"GestureBind v{version}",
        "CHANGELOG.md": f"## v{version}",
    }
    for filename, marker in contracts.items():
        try:
            content = (root / filename).read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"unable to read {filename}: {exc}")
            continue
        if marker not in content:
            errors.append(
                f"release version mismatch: {filename} does not contain {marker!r}"
            )
    return errors


def main() -> int:
    try:
        paths = tracked_files()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[repo-hygiene] unable to read git index: {exc}", file=sys.stderr)
        return 2

    errors = validate_tracked_files(paths)
    errors.extend(validate_release_metadata())
    if errors:
        print("[repo-hygiene] FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "[repo-hygiene] OK: "
        f"{len(paths)} tracked files, {len(RELEASE_MODEL_FILES)} release model files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
