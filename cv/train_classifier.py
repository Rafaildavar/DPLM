import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# --------------------------------------------------
# Тренер KNN: загрузка NPY семплов и обучение модели
# Комментарии на русском
# --------------------------------------------------


def load_dataset(data_root: Path, expect_dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Загружает семплы из data_root/<label>/sample_*.npy
    Возвращает (X, y, classes), где:
      - X: (N, D) — усреднённые по времени признаки семплов
      - y: (N,) — индексы классов
      - classes: список имён классов по индексу
    """
    # Временно храним признаки переменной длины, затем выровняем по max/expect_dim
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
            arr = np.load(sf)  # ожидаем (T, D1, D2) или (T, D)

            # Приводим к (T, D)
            if arr.ndim == 3:
                T, D1, D2 = arr.shape
                arr = arr.reshape(T, D1 * D2)
            elif arr.ndim != 2:
                print(f"[!] Неожиданная форма {sf}: {arr.shape}, пропуск")
                continue

            # Простая агрегация по времени: усреднение -> (D,)
            feat = arr.mean(axis=0)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение KNN классификатора жестов")
    p.add_argument("--data-root", default="data/gestures", help="Корень датасета")
    p.add_argument("--out", default="models/knn.pkl", help="Путь для сохранения модели")
    p.add_argument("--neighbors", type=int, default=5, help="Число соседей KNN")
    p.add_argument("--expect-dim", type=int, default=None, help="Ожидаемая длина признака (например, 42 или 84)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, y, classes = load_dataset(data_root, expect_dim=args.expect_dim)
    print(f"[i] Загружено семплов: {len(X)}; классов: {len(classes)}; размер признака: {X.shape[1]}")

    clf = KNeighborsClassifier(n_neighbors=args.neighbors, metric="euclidean")
    clf.fit(X, y)

    joblib.dump(clf, out_path)
    print(f"[✓] Модель сохранена: {out_path}")

    # Сохраним классы и размерность признака для инференса
    (out_path.parent / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2))
    (out_path.parent / "feature_dim.txt").write_text(str(X.shape[1]))
    print("[✓] Метаданные сохранены: classes.json, feature_dim.txt")


if __name__ == "__main__":
    main()


