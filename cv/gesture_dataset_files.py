"""Helpers for locating and grouping gesture sample files."""
from __future__ import annotations

import json
import re
from pathlib import Path

AUGMENTED_SAMPLE_PREFIX = "aug_sample_"
REAL_SAMPLE_PATTERN = "sample_*.npy"
AUGMENTED_SAMPLE_PATTERN = f"{AUGMENTED_SAMPLE_PREFIX}*.npy"

_REAL_INDEX_RE = re.compile(r"^sample_(?:auto_)?(?P<index>\d+)$")
_AUG_INDEX_RE = re.compile(
    r"^aug_sample_(?P<source>\d+)_(?P<variant>\d+)$"
)


def gesture_sample_paths(
    label_dir: Path,
    *,
    include_augmented: bool = False,
) -> list[Path]:
    """Return gesture sample paths in stable order.

    Positive synthetic augmentations are disabled by default because live tests
    showed that they can distort personalized gesture boundaries. Old
    ``aug_sample_*`` files may remain on disk, but training/sync code should use
    real camera samples unless an experiment explicitly opts in.
    """
    paths = set(label_dir.glob(REAL_SAMPLE_PATTERN))
    if include_augmented:
        paths.update(label_dir.glob(AUGMENTED_SAMPLE_PATTERN))
    return sorted(paths, key=sample_sort_key)


def real_sample_paths(label_dir: Path) -> list[Path]:
    """Return camera/generated samples, excluding positive augmentations."""
    return sorted(label_dir.glob(REAL_SAMPLE_PATTERN), key=sample_sort_key)


def augmented_sample_paths(label_dir: Path) -> list[Path]:
    return sorted(label_dir.glob(AUGMENTED_SAMPLE_PATTERN), key=sample_sort_key)


def is_augmented_sample_path(path: Path) -> bool:
    return path.name.startswith(AUGMENTED_SAMPLE_PREFIX)


def sample_group_key(path: Path) -> str:
    """Return the original capture key shared by a sample and its augmentations."""
    source_name = ""
    if is_augmented_sample_path(path):
        metadata_path = path.with_suffix(".meta.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_name = Path(str(metadata.get("source_sample") or "")).name
        except (OSError, ValueError, json.JSONDecodeError):
            source_name = ""
        if not source_name:
            match = _AUG_INDEX_RE.match(path.stem)
            if match:
                source_name = f"sample_{int(match.group('source')):04d}.npy"
    return str(path.parent / (source_name or path.name))


def sample_source_from_path(path: Path) -> str:
    if is_augmented_sample_path(path):
        return "augmented"
    if path.name.startswith("sample_auto_"):
        return "generated"
    return "dataset"


def augmented_sample_path(source_path: Path, variant_index: int) -> Path:
    source_index = _source_index_from_sample_path(source_path)
    return source_path.with_name(
        f"{AUGMENTED_SAMPLE_PREFIX}{source_index:04d}_{max(0, int(variant_index)):02d}.npy"
    )


def stable_sample_index(path: Path) -> int:
    """Return a DB-safe sample index.

    Existing camera files keep their historical index. Augmented samples are
    moved into a high range so ``sample_0000.npy`` and
    ``aug_sample_0000_00.npy`` can coexist under the same gesture.
    """
    stem = path.stem
    aug = _AUG_INDEX_RE.match(stem)
    if aug:
        source = int(aug.group("source"))
        variant = int(aug.group("variant"))
        return 1_000_000 + source * 100 + variant

    real = _REAL_INDEX_RE.match(stem)
    if real:
        return int(real.group("index"))

    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def sample_sort_key(path: Path) -> tuple[int, int, int, str]:
    stem = path.stem
    aug = _AUG_INDEX_RE.match(stem)
    if aug:
        return (1, int(aug.group("source")), int(aug.group("variant")), path.name)

    real = _REAL_INDEX_RE.match(stem)
    if real:
        kind = 2 if stem.startswith("sample_auto_") else 0
        return (kind, int(real.group("index")), -1, path.name)

    return (9, 0, 0, path.name)


def _source_index_from_sample_path(path: Path) -> int:
    stem = path.stem
    real = _REAL_INDEX_RE.match(stem)
    if real:
        return int(real.group("index"))
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0
