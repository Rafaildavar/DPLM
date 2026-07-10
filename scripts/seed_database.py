#!/usr/bin/env python3
"""Seed a reproducible GestureBind demo database.

The recognition pipeline is file-based: ``models/knn.pkl`` predicts class
indices and ``models/classes.json`` maps those indices to gesture labels. The
database then binds those labels to user commands. This script makes a fresh DB
usable by syncing PostgreSQL/SQLite rows to the current model metadata.

Run from the project root:

    python -m scripts.seed_database

The script is idempotent: repeated runs update the same demo rows and do not
create duplicate gestures, commands, or active model records.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from sqlalchemy.orm import Session

from app.models.database import (
    Command,
    Gesture,
    GestureSample,
    RecognitionModel,
    Settings,
    User,
    get_db_session,
    init_database,
)
from app.services.binding_settings import (
    KEY_CONFIDENCE_THRESHOLD,
    KEY_COOLDOWN_MS,
    KEY_WARN_TWO_HANDS,
)

ROOT = Path(__file__).resolve().parents[1]
GESTURES_DIR = ROOT / "data" / "gestures"
MODELS_DIR = ROOT / "models"
ACTIVE_MODEL_NAME = "knn-demo-current"


@dataclass(frozen=True)
class DemoBinding:
    label: str
    command_name: str
    description: str
    action_spec: dict[str, Any]
    platform: str = "macos"


DEMO_BINDINGS: tuple[DemoBinding, ...] = (
    DemoBinding(
        label="new",
        command_name="Open telegram",
        description="Демо-команда: открыть Telegram по жесту new.",
        action_spec={"action": "open_app", "app": "Telegram", "platform": "macos"},
    ),
    DemoBinding(
        label="new2",
        command_name="open browser S",
        description="Демо-команда: открыть Safari по жесту new2.",
        action_spec={"action": "open_app", "app": "Safari", "platform": "macos"},
    ),
)


DEFAULT_COMMANDS: tuple[dict[str, Any], ...] = (
    {
        "name": "Открыть Safari",
        "description": "Запускает браузер Safari.",
        "platform": "macos",
        "action_spec": {"action": "open_app", "app": "Safari", "platform": "macos"},
    },
    {
        "name": "Прокрутка вниз",
        "description": "Прокручивает активное окно вниз.",
        "platform": "all",
        "action_spec": {"action": "scroll", "clicks": -5, "platform": "all"},
    },
    {
        "name": "Громче",
        "description": "Увеличивает громкость системы.",
        "platform": "macos",
        "action_spec": {"action": "volume_up", "platform": "macos"},
    },
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _write_setting(session: Session, key: str, value: str) -> bool:
    row = session.query(Settings).filter(Settings.key == key).first()
    if row is None:
        session.add(Settings(key=key, value=value))
        return True
    if row.value != value:
        row.value = value
        return True
    return False


def _json_dump(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def load_model_classes(models_dir: Path = MODELS_DIR) -> list[str]:
    classes_path = models_dir / "classes.json"
    if not classes_path.exists():
        return []
    raw = json.loads(classes_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("models/classes.json должен быть JSON-списком")

    labels: list[str] = []
    seen: set[str] = set()
    for item in raw:
        label = str(item).strip()
        key = label.lower()
        if not label or key in seen:
            continue
        labels.append(label)
        seen.add(key)
    return labels


def load_feature_dim(models_dir: Path = MODELS_DIR) -> Optional[int]:
    path = models_dir / "feature_dim.txt"
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def find_samples_dir(label: str, gestures_dir: Path = GESTURES_DIR) -> Optional[Path]:
    if not gestures_dir.exists():
        return None

    candidates = sorted(p for p in gestures_dir.iterdir() if p.is_dir())
    for candidate in candidates:
        if candidate.name == label:
            return candidate

    lower = label.lower()
    for candidate in candidates:
        if candidate.name.lower() == lower:
            return candidate
    return None


def count_sample_files(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    return len(sorted(path.glob("sample_*.npy")))


def _sample_index(path: Path, fallback: int) -> int:
    raw = path.stem.removeprefix("sample_")
    try:
        return int(raw)
    except ValueError:
        return fallback


def ensure_default_user(session: Session) -> User:
    user = session.query(User).filter_by(username="default").first()
    if user is None:
        user = User(username="default", display_name="Локальный пользователь")
        session.add(user)
        session.flush()
    return user


def sync_gestures_with_model(
    session: Session,
    owner: User,
    classes: Iterable[str],
    *,
    gestures_dir: Path = GESTURES_DIR,
    clear_stale_class_ids: bool = True,
) -> dict[str, int]:
    labels = [str(label).strip() for label in classes if str(label).strip()]
    class_map = {label: idx for idx, label in enumerate(labels)}
    class_keys = {label.lower() for label in labels}

    summary = {"created": 0, "updated": 0, "stale": 0, "samples": 0}

    if clear_stale_class_ids:
        for row in session.query(Gesture).filter(Gesture.model_class_id.isnot(None)).all():
            if str(row.label or "").lower() not in class_keys:
                row.model_class_id = None
                summary["stale"] += 1

    for label, class_id in class_map.items():
        samples_dir = find_samples_dir(label, gestures_dir)
        samples_path = _rel(samples_dir) if samples_dir is not None else None
        row = session.query(Gesture).filter(Gesture.label == label).first()
        if row is None:
            row = Gesture(
                label=label,
                description=f"Демо-жест из текущей модели: {label}",
                samples_path=samples_path,
                model_class_id=class_id,
                accuracy=None,
                is_two_hands=False,
                is_active=True,
                user=owner,
            )
            session.add(row)
            summary["created"] += 1
        else:
            row.description = row.description or f"Демо-жест из текущей модели: {label}"
            row.samples_path = samples_path or row.samples_path
            row.model_class_id = class_id
            row.is_active = True
            if row.user_id is None:
                row.user = owner
            summary["updated"] += 1

    session.flush()

    for label in labels:
        row = session.query(Gesture).filter(Gesture.label == label).first()
        samples_dir = find_samples_dir(label, gestures_dir)
        if row is None or samples_dir is None:
            continue
        for fallback, sample_path in enumerate(sorted(samples_dir.glob("sample_*.npy"))):
            index = _sample_index(sample_path, fallback)
            sample = (
                session.query(GestureSample)
                .filter(
                    GestureSample.gesture_id == row.id,
                    GestureSample.sample_index == index,
                )
                .first()
            )
            if sample is None:
                sample = GestureSample(
                    gesture_id=row.id,
                    sample_index=index,
                    source="import",
                )
                session.add(sample)
                summary["samples"] += 1
            sample.features_path = _rel(sample_path)
            sample.source = "import"

    return summary


def register_model_metadata(
    session: Session,
    classes: list[str],
    *,
    models_dir: Path = MODELS_DIR,
    gestures_dir: Path = GESTURES_DIR,
) -> RecognitionModel:
    model_path = models_dir / "knn.pkl"
    classes_path = models_dir / "classes.json"
    feature_dim = load_feature_dim(models_dir)
    n_samples = sum(count_sample_files(find_samples_dir(label, gestures_dir)) for label in classes)

    session.query(RecognitionModel).filter(RecognitionModel.is_active.is_(True)).update(
        {RecognitionModel.is_active: False},
        synchronize_session=False,
    )

    row = session.query(RecognitionModel).filter_by(name=ACTIVE_MODEL_NAME).first()
    if row is None:
        row = RecognitionModel(name=ACTIVE_MODEL_NAME, algorithm="knn")
        session.add(row)

    row.algorithm = "knn"
    row.model_path = _rel(model_path) if model_path.exists() else None
    row.classes_path = _rel(classes_path) if classes_path.exists() else None
    row.feature_dim = feature_dim
    row.n_classes = len(classes)
    row.n_samples = n_samples
    row.hyperparams_json = _json_dump(
        {
            "source": "scripts.seed_database",
            "classes": classes,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    )
    row.is_active = True
    session.flush()
    return row


def ensure_command_binding(session: Session, binding: DemoBinding) -> bool:
    gesture = session.query(Gesture).filter(Gesture.label == binding.label).first()
    if gesture is None:
        return False

    existing_for_gesture = session.query(Command).filter(Command.gesture_id == gesture.id).first()
    row = session.query(Command).filter(Command.name == binding.command_name).first()
    if existing_for_gesture is not None and existing_for_gesture is not row:
        existing_for_gesture.gesture_id = None

    created = row is None
    if row is None:
        row = Command(name=binding.command_name)
        session.add(row)

    row.description = binding.description
    row.platform = binding.platform
    row.action_spec = _json_dump(binding.action_spec)
    row.script_path = None
    row.gesture_id = gesture.id
    row.is_active = True
    session.flush()
    return created


def ensure_demo_bindings(session: Session, classes: Iterable[str]) -> dict[str, int]:
    class_keys = {str(label).strip().lower() for label in classes}
    summary = {"created": 0, "updated": 0, "skipped": 0}
    for binding in DEMO_BINDINGS:
        if binding.label.lower() not in class_keys:
            summary["skipped"] += 1
            continue
        created = ensure_command_binding(session, binding)
        if created:
            summary["created"] += 1
        else:
            summary["updated"] += 1
    return summary


def ensure_default_commands(session: Session) -> int:
    added = 0
    for spec in DEFAULT_COMMANDS:
        row = session.query(Command).filter_by(name=spec["name"]).first()
        if row is None:
            row = Command(name=spec["name"])
            session.add(row)
            added += 1
        row.description = spec["description"]
        row.platform = spec["platform"]
        row.action_spec = _json_dump(spec["action_spec"])
        row.is_active = True
    session.flush()
    return added


def ensure_default_settings(session: Session) -> int:
    defaults = {
        "camera_index": "0",
        "smoothing_window": "30",
        "tts_enabled": "false",
        "tts_language": "ru",
        KEY_CONFIDENCE_THRESHOLD: "0.6500",
        KEY_COOLDOWN_MS: "1500",
        KEY_WARN_TWO_HANDS: "true",
    }
    changed = 0
    for key, value in defaults.items():
        if _write_setting(session, key, value):
            changed += 1
    session.flush()
    return changed


def seed_session(
    session: Session,
    *,
    gestures_dir: Path = GESTURES_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    classes = load_model_classes(models_dir)
    if not classes:
        raise RuntimeError(
            "models/classes.json пуст или отсутствует; сначала обучите модель"
        )

    owner = ensure_default_user(session)
    gestures = sync_gestures_with_model(
        session,
        owner,
        classes,
        gestures_dir=gestures_dir,
    )
    model = register_model_metadata(
        session,
        classes,
        models_dir=models_dir,
        gestures_dir=gestures_dir,
    )
    bindings = ensure_demo_bindings(session, classes)
    default_commands = ensure_default_commands(session)
    settings = ensure_default_settings(session)
    session.commit()

    return {
        "classes": classes,
        "gestures": gestures,
        "model": {
            "name": model.name,
            "feature_dim": model.feature_dim,
            "n_classes": model.n_classes,
            "n_samples": model.n_samples,
        },
        "bindings": bindings,
        "default_commands": default_commands,
        "settings": settings,
    }


def main() -> None:
    init_database()
    session = get_db_session()
    try:
        summary = seed_session(session)
    finally:
        session.close()

    print("[OK] Seed завершён")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
