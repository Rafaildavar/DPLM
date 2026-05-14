#!/usr/bin/env python3
"""Заполнение БД начальными данными.

Скрипт:

1. Создаёт пользователя по умолчанию (``default``).
2. Импортирует жесты из каталога ``data/gestures/<label>/`` (если есть).
3. Регистрирует обученную модель из ``models/`` (если файлы найдены).
4. Создаёт типовые команды (open Safari, scroll и т.п.) — только если их нет.

Запуск:

    python -m scripts.seed_database

Скрипт безопасно перезапускается: повторный запуск не плодит дубликатов.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

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

ROOT = Path(__file__).resolve().parents[1]
GESTURES_DIR = ROOT / "data" / "gestures"
MODELS_DIR = ROOT / "models"


def ensure_default_user(session) -> User:
    user = session.query(User).filter_by(username="default").first()
    if user is None:
        user = User(username="default", display_name="Локальный пользователь")
        session.add(user)
        session.commit()
        print(f"[+] Создан пользователь: {user}")
    return user


def import_gestures_from_disk(session, owner: User) -> int:
    if not GESTURES_DIR.exists():
        print(f"[i] Каталог {GESTURES_DIR} не найден — импорт пропущен")
        return 0

    added = 0
    for sub in sorted(p for p in GESTURES_DIR.iterdir() if p.is_dir()):
        label = sub.name
        gesture = session.query(Gesture).filter_by(label=label).first()
        if gesture is None:
            gesture = Gesture(
                label=label,
                samples_path=str(sub.relative_to(ROOT)),
                user=owner,
                description=f"Автоимпорт из {sub.relative_to(ROOT)}",
                is_two_hands=label.lower().endswith("_two")
                or "two" in label.lower(),
            )
            session.add(gesture)
            session.flush()
            added += 1
            print(f"[+] Импортирован жест: {label}")

        # Регистрируем образцы по файлам (если есть).
        sample_files = sorted(
            list(sub.glob("*.npy")) + list(sub.glob("*.json"))
        )
        for idx, fp in enumerate(sample_files, start=1):
            already = (
                session.query(GestureSample)
                .filter_by(gesture_id=gesture.id, sample_index=idx)
                .first()
            )
            if already is not None:
                continue
            session.add(
                GestureSample(
                    gesture_id=gesture.id,
                    sample_index=idx,
                    features_path=str(fp.relative_to(ROOT)),
                    source="import",
                )
            )

    session.commit()
    return added


def register_models(session) -> Optional[RecognitionModel]:
    knn = MODELS_DIR / "knn.pkl"
    classes = MODELS_DIR / "classes.json"
    feat_dim_file = MODELS_DIR / "feature_dim.txt"
    if not knn.exists():
        print(f"[i] {knn} не найден — регистрация модели пропущена")
        return None

    name = f"knn-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    existing_active = (
        session.query(RecognitionModel).filter_by(is_active=True).first()
    )

    feature_dim = None
    if feat_dim_file.exists():
        try:
            feature_dim = int(feat_dim_file.read_text().strip())
        except ValueError:
            feature_dim = None

    n_classes = None
    if classes.exists():
        try:
            n_classes = len(json.loads(classes.read_text()))
        except Exception:
            n_classes = None

    model = RecognitionModel(
        name=name,
        algorithm="knn",
        model_path=str(knn.relative_to(ROOT)),
        classes_path=str(classes.relative_to(ROOT)) if classes.exists() else None,
        feature_dim=feature_dim,
        n_classes=n_classes,
        is_active=existing_active is None,
    )
    session.add(model)
    session.commit()
    print(f"[+] Зарегистрирована модель: {model}")
    return model


def ensure_default_commands(session) -> int:
    samples = [
        {
            "name": "Открыть Safari",
            "description": "Запускает браузер Safari (macOS).",
            "platform": "macos",
            "action_spec": json.dumps(
                {"action": "open_app", "app": "Safari"}, ensure_ascii=False
            ),
        },
        {
            "name": "Прокрутка вниз",
            "description": "Прокрутка активного окна на 5 шагов вниз.",
            "platform": "all",
            "action_spec": json.dumps(
                {"action": "scroll", "clicks": -5}, ensure_ascii=False
            ),
        },
        {
            "name": "Громче",
            "description": "Увеличить громкость системы.",
            "platform": "macos",
            "action_spec": json.dumps(
                {"action": "volume_up"}, ensure_ascii=False
            ),
        },
    ]
    added = 0
    for spec in samples:
        if session.query(Command).filter_by(name=spec["name"]).first():
            continue
        session.add(Command(**spec))
        added += 1
    session.commit()
    if added:
        print(f"[+] Добавлено типовых команд: {added}")
    return added


def ensure_default_settings(session) -> None:
    defaults = {
        "camera_index": "0",
        "smoothing_window": "30",
        "tts_enabled": "true",
        "tts_language": "ru",
    }
    for key, value in defaults.items():
        if not session.query(Settings).filter_by(key=key).first():
            session.add(Settings(key=key, value=value))
    session.commit()


def main() -> None:
    init_database()
    session = get_db_session()
    try:
        user = ensure_default_user(session)
        n_g = import_gestures_from_disk(session, user)
        register_models(session)
        n_c = ensure_default_commands(session)
        ensure_default_settings(session)
        print(
            f"[OK] Seed завершён. Новых жестов: {n_g}, новых команд: {n_c}"
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
