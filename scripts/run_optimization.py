import optuna
import numpy as np
import joblib
import os
import json
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def load_dataset(data_root: Path):
    X_list = []
    y_list = []
    classes = []
    target_dim = 42
    
    label_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    for idx, label_dir in enumerate(label_dirs):
        classes.append(label_dir.name)
        sample_files = list(label_dir.glob("sample_*.npy"))
        for sf in sample_files:
            arr = np.load(sf)
            if arr.ndim == 3:
                T, D1, D2 = arr.shape
                arr = arr.reshape(T, D1 * D2)
            
            # Усреднение по времени
            feat = arr.mean(axis=0)
            
            # Выравнивание размерности до 42
            if feat.shape[0] > target_dim:
                feat = feat[:target_dim]
            elif feat.shape[0] < target_dim:
                pad = np.zeros(target_dim - feat.shape[0])
                feat = np.concatenate([feat, pad])
                
            X_list.append(feat)
            y_list.append(idx)
            
    return np.array(X_list), np.array(y_list), classes

def objective(trial, X, y):
    k = trial.suggest_int('n_neighbors', 1, 15, step=2)
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev'])
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
    score = cross_val_score(clf, X, y, cv=3, scoring='accuracy').mean()
    return score

def main():
    data_root = Path("data/gestures")
    if not data_root.exists():
        print("Ошибка: папка с жестами не найдена.")
        return

    X, y, classes = load_dataset(data_root)
    print(f"Загружено {len(X)} образцов для {len(classes)} классов.")

    # Создание исследования
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    print("\n[✓] Оптимизация завершена!")
    print(f"Лучшая точность: {study.best_value:.4f}")
    print(f"Лучшие параметры: {study.best_params}")

    # Сохранение графиков через Matplotlib
    docs_dir = Path("docs/plots")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. История оптимизации
    plt.figure(figsize=(10, 6))
    values = [t.value for t in study.trials if t.value is not None]
    plt.plot(values, marker='o', linestyle='-', color='b', label='Accuracy')
    best_values = np.maximum.accumulate(values)
    plt.plot(best_values, marker='', linestyle='--', color='r', label='Best Accuracy')
    plt.title('История оптимизации гиперпараметров')
    plt.xlabel('Итерация')
    plt.ylabel('Точность (F1-score)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(str(docs_dir / "opt_history.png"), dpi=300)
    plt.close()

    # 2. Важность параметров (упрощенно)
    import optuna.importance as importance
    try:
        importances = importance.get_param_importances(study)
        plt.figure(figsize=(10, 6))
        plt.bar(importances.keys(), importances.values(), color='skyblue')
        plt.title('Важность гиперпараметров')
        plt.ylabel('Относительная важность')
        plt.savefig(str(docs_dir / "param_importance.png"), dpi=300)
        plt.close()
    except:
        print("Не удалось рассчитать важность параметров.")

    print(f"\n[✓] Графики сохранены в папку {docs_dir}")

if __name__ == "__main__":
    main()
