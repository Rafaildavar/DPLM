import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Тренер KNN: загрузка NPY семплов и обучение модели
# Комментарии на русском
# --------------------------------------------------


def load_dataset(data_root: Path, expect_dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Загружает семплы из data_root/<label>/sample_*.npy и возвращает (X, y, classes)."""
    feats_raw: List[np.ndarray] = []
    y_list: List[int] = []
    classes: List[str] = []
    max_dim: int = 0

    for label_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        label = label_dir.name
        sample_files = sorted(label_dir.glob("sample_*.npy"))
        if not sample_files:
            print(f"[i] Пропуск: нет семплов в {label_dir}")
            continue

        class_idx = len(classes)
        classes.append(label)

        for sf in sample_files:
            arr = np.load(sf)
            if arr.ndim == 3:
                t, d1, d2 = arr.shape
                arr = arr.reshape(t, d1 * d2)
            elif arr.ndim != 2:
                print(f"[!] Неожиданная форма {sf}: {arr.shape}, пропуск")
                continue

            feat = arr.mean(axis=0)
            feats_raw.append(feat.astype(np.float32, copy=False))
            y_list.append(class_idx)
            max_dim = max(max_dim, int(feat.shape[0]))

    if not feats_raw:
        raise RuntimeError("Датасет пуст — не найдено ни одного семпла")

    target_dim = expect_dim if expect_dim is not None else max_dim
    if expect_dim is not None and max_dim > expect_dim:
        print(f"[w] Найдены признаки длиной {max_dim} > ожидаемой {expect_dim}. Лишние компоненты будут обрезаны.")

    x_aligned: List[np.ndarray] = []
    for feat in feats_raw:
        if feat.shape[0] > target_dim:
            x_aligned.append(feat[:target_dim])
        elif feat.shape[0] < target_dim:
            pad = np.zeros(target_dim - feat.shape[0], dtype=feat.dtype)
            x_aligned.append(np.concatenate([feat, pad], axis=0))
        else:
            x_aligned.append(feat)

    x = np.stack(x_aligned, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    return x, y, classes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение KNN классификатора жестов")
    p.add_argument("--data-root", default="data/gestures", help="Корень датасета")
    p.add_argument("--out", default="models/knn.pkl", help="Путь для сохранения модели")
    p.add_argument("--neighbors", type=int, default=5, help="Число соседей KNN")
    p.add_argument("--expect-dim", type=int, default=None, help="Ожидаемая длина признака")
    p.add_argument("--eval-size", type=float, default=0.2, help="Доля тестовой выборки")
    p.add_argument("--random-state", type=int, default=42, help="Сид разделения train/test")
    p.add_argument("--metrics-out", default="models/train_metrics.json", help="JSON-файл с метриками")
    return p.parse_args()


def build_model(neighbors: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=neighbors, metric="euclidean", weights="distance")),
        ]
    )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x, y, classes = load_dataset(data_root, expect_dim=args.expect_dim)
    print(f"[i] Загружено семплов: {len(x)}; классов: {len(classes)}; размер признака: {x.shape[1]}")

    model = build_model(args.neighbors)

    metrics = {"train_samples": int(len(x)), "feature_dim": int(x.shape[1]), "classes": classes}
    if len(np.unique(y)) > 1 and len(x) >= 5:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.eval_size,
            random_state=args.random_state,
            stratify=y,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics.update(
            {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "test_samples": int(len(x_test)),
            }
        )
        print(f"[i] holdout accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    else:
        print("[w] Недостаточно данных для корректной валидации — метрики holdout пропущены")

    model.fit(x, y)
    joblib.dump(model, out_path)
    print(f"[✓] Модель сохранена: {out_path}")

    (out_path.parent / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2))
    (out_path.parent / "feature_dim.txt").write_text(str(x.shape[1]))

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[✓] Метаданные сохранены: classes.json, feature_dim.txt, {metrics_path}")


if __name__ == "__main__":
    main()
