"""Failure-safe publication of a classifier and its inference sidecars."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib


def _temporary_sibling(target: Path, *, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def publish_model_bundle_atomic(
    estimator: Any,
    model_path: Path,
    text_sidecars: Mapping[Path, str],
) -> None:
    """Publish a complete model bundle with rollback and model-last activation.

    Every file is written to a sibling temporary file first. Sidecars are
    installed before the model, which acts as the activation marker. If any
    replacement fails, all already replaced files are restored.
    """
    model_target = Path(model_path)
    sidecars = [(Path(path), str(payload)) for path, payload in text_sidecars.items()]
    targets = [path for path, _payload in sidecars] + [model_target]
    if len(set(targets)) != len(targets):
        raise ValueError("model bundle contains duplicate target paths")

    temporary_files: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    installed: list[Path] = []
    try:
        for target, payload in sidecars:
            temporary = _temporary_sibling(target, suffix=".tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary_files[target] = temporary

        model_temporary = _temporary_sibling(model_target, suffix=".tmp")
        joblib.dump(estimator, model_temporary)
        temporary_files[model_target] = model_temporary

        for target in targets:
            backup: Path | None = None
            if target.exists():
                backup = _temporary_sibling(target, suffix=".bak")
                backup.unlink()
                os.replace(target, backup)
            backups[target] = backup
            try:
                os.replace(temporary_files[target], target)
            except Exception:
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                    backups[target] = None
                raise
            installed.append(target)

    except Exception:
        for target in reversed(installed):
            if target.exists():
                target.unlink()
            backup = backups.get(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
                backups[target] = None
        raise
    finally:
        for temporary in temporary_files.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            if backup is not None:
                backup.unlink(missing_ok=True)
