import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cv.gesture_features import (
    FEATURE_STATIC_MEAN,
    SUPPORTED_FEATURE_MODES,
    build_feature_vector,
)

SUPPORTED_MODEL_TYPES = ("knn", "svm", "extra_trees", "rf", "logreg")


# --------------------------------------------------
# Тренер KNN: загрузка NPY семплов и обучение модели
# Комментарии на русском
# --------------------------------------------------


def load_dataset(
    data_root: Path,
    expect_dim: Optional[int] = None,
    include_labels: Optional[Iterable[str]] = None,
    lowercase_labels: bool = False,
    feature_mode: str = FEATURE_STATIC_MEAN,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Загружает семплы из data_root/<label>/sample_*.npy
    Возвращает (X, y, classes), где:
      - X: (N, D) — признаки семплов в выбранном feature_mode
      - y: (N,) — индексы классов
      - classes: список имён классов по индексу
    """
    # Временно храним признаки переменной длины, затем выровняем по max/expect_dim
    feats_raw: List[np.ndarray] = []
    y_list: List[int] = []
    classes: List[str] = []
    class_indices: dict[str, int] = {}
    max_dim: int = 0
    raw_include = {str(label).strip() for label in (include_labels or []) if str(label).strip()}
    canonical_include = {
        label.lower() if lowercase_labels else label for label in raw_include
    }

    for label_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        raw_label = label_dir.name
        label = raw_label.lower() if lowercase_labels else raw_label
        if canonical_include and label not in canonical_include and raw_label not in raw_include:
            continue
        sample_files = sorted(label_dir.glob("sample_*.npy"))
        if not sample_files:
            print(f"[i] Пропуск: нет семплов в {label_dir}")
            continue

        if label not in class_indices:
            class_indices[label] = len(classes)
            classes.append(label)
        class_idx = class_indices[label]

        for sf in sample_files:
            arr = np.load(sf)  # ожидаем (T, D1, D2) или (T, D)

            # Приводим к (T, D)
            if arr.ndim == 3:
                T, D1, D2 = arr.shape
                arr = arr.reshape(T, D1 * D2)
            elif arr.ndim != 2:
                print(f"[!] Неожиданная форма {sf}: {arr.shape}, пропуск")
                continue

            feat = build_feature_vector(arr, mode=feature_mode, target_dim=expect_dim)
            # Сохраняем как есть, выровняем позже
            feats_raw.append(feat.astype(np.float32, copy=False))
            y_list.append(class_idx)
            if feat.shape[0] > max_dim:
                max_dim = int(feat.shape[0])

    if not feats_raw:
        raise RuntimeError("Датасет пуст — не найдено ни одного семпла")

    # Определяем целевую размерность признака
    target_dim = expect_dim if expect_dim is not None else max_dim
    if expect_dim is not None and max_dim > expect_dim:
        print(f"[w] Найдены признаки длиной {max_dim} > ожидаемой {expect_dim}. Лишние компоненты будут обрезаны.")
    if expect_dim is None and len(set(f.shape[0] for f in feats_raw)) > 1:
        print(f"[i] Выравниваем разные длины признаков до {target_dim} (дополнение нулями/обрезка)")

    # Выровнять признаки до target_dim: обрезать или дополнить нулями
    X_aligned: List[np.ndarray] = []
    for f in feats_raw:
        if f.shape[0] == target_dim:
            X_aligned.append(f)
        elif f.shape[0] > target_dim:
            X_aligned.append(f[:target_dim])
        else:
            pad = np.zeros(target_dim - f.shape[0], dtype=f.dtype)
            X_aligned.append(np.concatenate([f, pad], axis=0))

    X = np.stack(X_aligned, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, classes


def build_classifier(
    model_type: str,
    *,
    neighbors: int = 5,
    weights: str = "distance",
    random_state: int = 42,
):
    model = str(model_type or "knn").strip().lower()
    if model == "knn":
        return KNeighborsClassifier(
            n_neighbors=max(1, int(neighbors)),
            metric="euclidean",
            weights=weights,
        )
    if model == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=2.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=int(random_state),
            ),
        )
    if model == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=250,
            random_state=int(random_state),
            class_weight="balanced",
        )
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=int(random_state),
            class_weight="balanced",
        )
    if model == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=int(random_state),
            ),
        )
    raise ValueError(f"unsupported model type: {model_type}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение KNN классификатора жестов")
    p.add_argument("--data-root", default="data/gestures", help="Корень датасета")
    p.add_argument("--out", default="models/knn.pkl", help="Путь для сохранения модели")
    p.add_argument("--classes-out", default=None, help="Путь для сохранения classes.json")
    p.add_argument("--feature-dim-out", default=None, help="Путь для сохранения feature_dim.txt")
    p.add_argument("--feature-mode-out", default=None, help="Путь для сохранения feature_mode.txt")
    p.add_argument(
        "--feature-mode",
        choices=SUPPORTED_FEATURE_MODES,
        default=FEATURE_STATIC_MEAN,
        help="Режим признаков: static_mean — текущий production baseline; dynamic_stats/hybrid_stats — JMLC-эксперименты",
    )
    p.add_argument("--neighbors", type=int, default=5, help="Число соседей KNN")
    p.add_argument(
        "--model-type",
        choices=SUPPORTED_MODEL_TYPES,
        default="knn",
        help="Тип классификатора: knn, svm, extra_trees, rf или logreg",
    )
    p.add_argument("--random-state", type=int, default=42, help="Seed для моделей с рандомизацией")
    p.add_argument(
        "--weights",
        choices=["uniform", "distance"],
        default="distance",
        help="Вес соседей KNN: distance устойчивее для маленьких несбалансированных наборов",
    )
    p.add_argument("--expect-dim", type=int, default=None, help="Ожидаемая длина признака (например, 42 или 84)")
    p.add_argument(
        "--include-label",
        action="append",
        default=[],
        help="Обучать только указанный класс; можно передать несколько раз",
    )
    p.add_argument(
        "--lowercase-labels",
        action="store_true",
        help="Сохранять имена классов в нижнем регистре (New -> new)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, y, classes = load_dataset(
        data_root,
        expect_dim=args.expect_dim,
        include_labels=args.include_label,
        lowercase_labels=bool(args.lowercase_labels),
        feature_mode=str(args.feature_mode),
    )
    print(
        f"[i] Загружено семплов: {len(X)}; классов: {len(classes)}; "
        f"режим признаков: {args.feature_mode}; размер признака: {X.shape[1]}; "
        f"модель: {args.model_type}"
    )

    clf = build_classifier(
        str(args.model_type),
        neighbors=int(args.neighbors),
        weights=str(args.weights),
        random_state=int(args.random_state),
    )
    clf.fit(X, y)

    joblib.dump(clf, out_path)
    print(f"[✓] Модель сохранена: {out_path}")

    # Сохраним классы и размерность признака для инференса
    classes_out = Path(args.classes_out) if args.classes_out else (out_path.parent / "classes.json")
    feature_dim_out = (
        Path(args.feature_dim_out)
        if args.feature_dim_out
        else (out_path.parent / "feature_dim.txt")
    )
    feature_mode_out = (
        Path(args.feature_mode_out)
        if args.feature_mode_out
        else (out_path.parent / "feature_mode.txt")
    )
    classes_out.parent.mkdir(parents=True, exist_ok=True)
    feature_dim_out.parent.mkdir(parents=True, exist_ok=True)
    feature_mode_out.parent.mkdir(parents=True, exist_ok=True)
    classes_out.write_text(json.dumps(classes, ensure_ascii=False, indent=2))
    feature_dim_out.write_text(str(X.shape[1]))
    feature_mode_out.write_text(str(args.feature_mode))
    print(f"[✓] Метаданные сохранены: {classes_out}, {feature_dim_out}, {feature_mode_out}")


if __name__ == "__main__":
    main()
