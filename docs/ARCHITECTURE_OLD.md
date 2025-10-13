# Архитектура старой версии (v0.6.0-old)

## Обзор

Старая версия представляла собой базовую систему распознавания жестов с возможностью обучения пользовательским жестам и классификацией в реальном времени. Система состояла из независимых CLI-модулей и простого GUI на PySide6 (Qt Widgets).

## Структура проекта

```
DPLM/
├── app/
│   ├── gui_main.py          # GUI на PySide6 Qt Widgets
│   └── data/gestures/       # Локальные данные жестов (дубликат)
├── cv/
│   ├── realtime_hands.py    # Детекция одной руки (MediaPipe)
│   ├── realtime_hands_dual.py  # Детекция двух рук
│   ├── record_gestures.py   # CLI: запись семплов жестов
│   ├── train_classifier.py  # CLI: обучение KNN модели
│   └── realtime_infer.py    # CLI: инференс в реальном времени
├── data/gestures/           # Датасет: <label>/sample_*.npy
├── models/
│   ├── knn.pkl              # Обученная KNN модель
│   ├── classes.json         # Маппинг классов
│   └── feature_dim.txt      # Размерность признака
├── docs/                    # Отчёты по этапам (txt)
└── scripts/                 # Вспомогательные скрипты
```

## Компоненты

### 1. Computer Vision (cv/)

#### 1.1 realtime_hands.py
- **Назначение**: Базовый модуль детекции руки через MediaPipe Hands
- **Функциональность**:
  - Захват видео с веб-камеры (AVFoundation backend для macOS M1)
  - Детекция одной руки (max_num_hands=1)
  - Извлечение 21 ландмарка (x, y координаты)
  - Нормализация `wrist_scale`: центрирование по запястью, масштабирование по максимальной L2-дистанции
  - Отрисовка скелета руки на кадре
  - Вывод нормализованных координат в консоль
- **Технологии**: OpenCV, MediaPipe Hands

#### 1.2 realtime_hands_dual.py
- **Назначение**: Расширение для детекции двух рук одновременно
- **Отличия от базового**:
  - `max_num_hands=2` в MediaPipe Hands
  - Обработка `results.multi_handedness` для определения "Left/Right"
  - Отображение меток рук на кадре

#### 1.3 record_gestures.py
- **Назначение**: CLI-инструмент для записи семплов жестов
- **Аргументы**:
  - `--label`: имя жеста (создаёт папку `data/gestures/<label>/`)
  - `--num-samples`: количество семплов для записи (по умолчанию 20)
  - `--frames`: длина одного семпла в кадрах (по умолчанию 30)
  - `--two-hands`: флаг для режима двух рук
- **Процесс**:
  1. Пользователь нажимает `s` для старта/стоп записи
  2. Накапливается буфер кадров (нормализованные ландмарки)
  3. Нажатие `n` сохраняет буфер как `sample_XXXX.npy`
  4. Формат: одна рука (21×2), две руки (42×2), развёрнуто по времени (T, D)
- **Выход**: NPY-файлы в `data/gestures/<label>/`

#### 1.4 train_classifier.py
- **Назначение**: Обучение KNN классификатора на записанных семплах
- **Аргументы**:
  - `--data-root`: путь к корню датасета (по умолчанию `data/gestures`)
  - `--out`: путь для сохранения модели (по умолчанию `models/knn.pkl`)
  - `--neighbors`: число соседей KNN (по умолчанию 5)
  - `--expect-dim`: ожидаемая размерность признака (опционально, для выравнивания)
- **Процесс**:
  1. Загрузка всех NPY-файлов из подпапок (каждая подпапка = класс)
  2. Приведение к единой размерности (padding/truncation)
  3. Агрегация по времени: `arr.mean(axis=0)` → вектор признаков (D,)
  4. Обучение KNeighborsClassifier (sklearn)
  5. Сохранение модели (`joblib.dump`)
  6. Сохранение метаданных: `classes.json`, `feature_dim.txt`

#### 1.5 realtime_infer.py
- **Назначение**: Инференс в реальном времени с опциональным TTS
- **Аргументы**:
  - `--model`: путь к модели (по умолчанию `models/knn.pkl`)
  - `--tts`: включить озвучку распознанных жестов (pyttsx3)
  - `--window`: размер окна сглаживания предсказаний (по умолчанию 30 кадров)
  - `--two-hands`: режим двух рук (определяется автоматически из `feature_dim.txt`)
- **Процесс**:
  1. Загрузка KNN модели, классов, feature_dim
  2. Захват видео, детекция руки(рук)
  3. Нормализация ландмарков
  4. Предсказание класса: `model.predict(features)`
  5. Сглаживание: мажоритарное голосование по последним N кадрам
  6. Опционально: озвучка жеста через pyttsx3
  7. Отображение предсказания на экране

### 2. GUI (app/gui_main.py)

#### Архитектура
- **Фреймворк**: PySide6 (Qt Widgets, не QML)
- **Главное окно**: `MainWindow(QMainWindow)`
  - Превью камеры: `CameraWidget` с QTimer (30 FPS)
  - Кнопки: "Старт инференса", "Запись жеста"
- **Диалоги**:
  - `RecordDialog`: ввод параметров для записи жеста (label, frames, samples, two-hands)
- **Интеграция с CV-модулями**:
  - Запуск через `subprocess.Popen()` отдельных Python-скриптов
  - Неблокирующее выполнение (GUI остаётся отзывчивым)

#### Проблемы старой архитектуры
1. **Отсутствие интеграции**: CV-модули запускаются как отдельные процессы, нет связи с GUI
2. **Нет системы команд**: жесты распознаются, но не привязаны к действиям
3. **Примитивный UI**: Qt Widgets без Material Design, минималистичности, прозрачности
4. **Отсутствие БД**: данные хранятся в файловой системе (NPY, JSON), нет централизованного управления
5. **Нет голосового помощника**: TTS есть только в CLI-модуле, нет STT
6. **Жёсткая привязка к файлам**: пути к скриптам захардкожены, нет абстракции

### 3. Хранение данных

#### 3.1 Семплы жестов
- **Формат**: NumPy NPY (binary)
- **Структура**: `data/gestures/<label>/sample_XXXX.npy`
- **Содержимое**: массивы формы (T, 21, 2) или (T, 42, 2) - временные последовательности нормализованных ландмарков

#### 3.2 Модели
- **knn.pkl**: сериализованный KNeighborsClassifier (joblib)
- **classes.json**: `["gesture1", "gesture2", ...]`
- **feature_dim.txt**: `"42"` или `"84"` (размерность признака)

#### 3.3 Отчёты
- **Формат**: TXT-файлы в `docs/report_stageX.txt`
- **Содержимое**: описание каждого этапа разработки (на русском)

### 4. Технологический стек

| Компонент | Библиотека/Технология | Версия |
|-----------|----------------------|--------|
| Computer Vision | MediaPipe Hands | latest |
| Видео захват | OpenCV (cv2) | latest |
| ML классификация | scikit-learn (KNN) | latest |
| GUI | PySide6 (Qt Widgets) | latest |
| TTS | pyttsx3 | latest |
| Сериализация | NumPy, joblib, json | - |
| Python | CPython | 3.10-3.13 |

### 5. Рабочий процесс пользователя

1. **Запись жеста**:
   - Запустить `python cv/record_gestures.py --label zoom --num-samples 30 --frames 30`
   - Нажать `s` для старта записи
   - Выполнить жест 30 раз
   - Нажать `n` для сохранения семпла
   - Повторить 30 раз

2. **Обучение модели**:
   - Запустить `python cv/train_classifier.py --data-root data/gestures`
   - Модель сохранена в `models/knn.pkl`

3. **Инференс**:
   - Запустить `python cv/realtime_infer.py --tts`
   - Выполнить жест перед камерой → система озвучивает название

4. **GUI-режим**:
   - Запустить `python -m app.gui_main`
   - Кликнуть "Запись жеста" → откроется диалог → запустится subprocess
   - Кликнуть "Старт инференса" → запустится subprocess с инференсом

### 6. Ключевые недостатки для редизайна

1. **Архитектурные**:
   - Отсутствие модульности: CV, GUI, ML - разрозненные скрипты
   - Нет сервисной архитектуры (всё CLI)
   - Subprocess-подход вместо интеграции

2. **Функциональные**:
   - Нет системы команд (gesture → action)
   - Нет БД для управления командами/жестами
   - Нет голосового помощника (STT отсутствует)
   - Нет кроссплатформенного исполнителя команд

3. **UI/UX**:
   - Примитивный интерфейс Qt Widgets
   - Нет Material Design 3
   - Нет минималистичной панели / аватара
   - Нет визуальной обратной связи при распознавании

4. **Данные**:
   - Файловое хранилище (NPY, JSON) - нет запросов, статистики
   - Нет истории использования жестов
   - Нет настроек пользователя

---

# Old Version Architecture (v0.6.0-old)

## Overview

The old version was a basic gesture recognition system with the ability to train custom gestures and perform real-time classification. The system consisted of independent CLI modules and a simple PySide6 (Qt Widgets) GUI.

## Project Structure

```
DPLM/
├── app/
│   ├── gui_main.py          # PySide6 Qt Widgets GUI
│   └── data/gestures/       # Local gesture data (duplicate)
├── cv/
│   ├── realtime_hands.py    # Single hand detection (MediaPipe)
│   ├── realtime_hands_dual.py  # Dual hand detection
│   ├── record_gestures.py   # CLI: record gesture samples
│   ├── train_classifier.py  # CLI: train KNN model
│   └── realtime_infer.py    # CLI: real-time inference
├── data/gestures/           # Dataset: <label>/sample_*.npy
├── models/
│   ├── knn.pkl              # Trained KNN model
│   ├── classes.json         # Class mapping
│   └── feature_dim.txt      # Feature dimension
├── docs/                    # Stage reports (txt)
└── scripts/                 # Helper scripts
```

## Components

### 1. Computer Vision (cv/)

#### 1.1 realtime_hands.py
- **Purpose**: Base module for hand detection via MediaPipe Hands
- **Functionality**:
  - Video capture from webcam (AVFoundation backend for macOS M1)
  - Single hand detection (max_num_hands=1)
  - Extraction of 21 landmarks (x, y coordinates)
  - `wrist_scale` normalization: center on wrist, scale by max L2-distance
  - Hand skeleton rendering on frame
  - Output normalized coordinates to console
- **Technologies**: OpenCV, MediaPipe Hands

#### 1.2 realtime_hands_dual.py
- **Purpose**: Extension for detecting two hands simultaneously
- **Differences from base**:
  - `max_num_hands=2` in MediaPipe Hands
  - Processing `results.multi_handedness` for "Left/Right" determination
  - Display hand labels on frame

#### 1.3 record_gestures.py
- **Purpose**: CLI tool for recording gesture samples
- **Arguments**:
  - `--label`: gesture name (creates folder `data/gestures/<label>/`)
  - `--num-samples`: number of samples to record (default 20)
  - `--frames`: sample length in frames (default 30)
  - `--two-hands`: flag for two-hand mode
- **Process**:
  1. User presses `s` to start/stop recording
  2. Buffer accumulates frames (normalized landmarks)
  3. Press `n` to save buffer as `sample_XXXX.npy`
  4. Format: one hand (21×2), two hands (42×2), unrolled over time (T, D)
- **Output**: NPY files in `data/gestures/<label>/`

#### 1.4 train_classifier.py
- **Purpose**: Train KNN classifier on recorded samples
- **Arguments**:
  - `--data-root`: dataset root path (default `data/gestures`)
  - `--out`: model save path (default `models/knn.pkl`)
  - `--neighbors`: number of KNN neighbors (default 5)
  - `--expect-dim`: expected feature dimension (optional, for alignment)
- **Process**:
  1. Load all NPY files from subfolders (each subfolder = class)
  2. Align to uniform dimension (padding/truncation)
  3. Temporal aggregation: `arr.mean(axis=0)` → feature vector (D,)
  4. Train KNeighborsClassifier (sklearn)
  5. Save model (`joblib.dump`)
  6. Save metadata: `classes.json`, `feature_dim.txt`

#### 1.5 realtime_infer.py
- **Purpose**: Real-time inference with optional TTS
- **Arguments**:
  - `--model`: model path (default `models/knn.pkl`)
  - `--tts`: enable gesture vocalization (pyttsx3)
  - `--window`: prediction smoothing window size (default 30 frames)
  - `--two-hands`: two-hand mode (auto-detected from `feature_dim.txt`)
- **Process**:
  1. Load KNN model, classes, feature_dim
  2. Video capture, hand(s) detection
  3. Landmark normalization
  4. Class prediction: `model.predict(features)`
  5. Smoothing: majority vote over last N frames
  6. Optional: gesture vocalization via pyttsx3
  7. Display prediction on screen

### 2. GUI (app/gui_main.py)

#### Architecture
- **Framework**: PySide6 (Qt Widgets, not QML)
- **Main Window**: `MainWindow(QMainWindow)`
  - Camera preview: `CameraWidget` with QTimer (30 FPS)
  - Buttons: "Start inference", "Record gesture"
- **Dialogs**:
  - `RecordDialog`: input parameters for gesture recording (label, frames, samples, two-hands)
- **CV Module Integration**:
  - Launch via `subprocess.Popen()` separate Python scripts
  - Non-blocking execution (GUI remains responsive)

#### Old Architecture Problems
1. **Lack of integration**: CV modules run as separate processes, no GUI connection
2. **No command system**: gestures recognized but not bound to actions
3. **Primitive UI**: Qt Widgets without Material Design, minimalism, transparency
4. **No database**: data stored in filesystem (NPY, JSON), no centralized management
5. **No voice assistant**: TTS only in CLI module, no STT
6. **Hard-coded file paths**: script paths hardcoded, no abstraction

### 3. Data Storage

#### 3.1 Gesture Samples
- **Format**: NumPy NPY (binary)
- **Structure**: `data/gestures/<label>/sample_XXXX.npy`
- **Content**: arrays of shape (T, 21, 2) or (T, 42, 2) - temporal sequences of normalized landmarks

#### 3.2 Models
- **knn.pkl**: serialized KNeighborsClassifier (joblib)
- **classes.json**: `["gesture1", "gesture2", ...]`
- **feature_dim.txt**: `"42"` or `"84"` (feature dimension)

#### 3.3 Reports
- **Format**: TXT files in `docs/report_stageX.txt`
- **Content**: description of each development stage (in Russian)

### 4. Technology Stack

| Component | Library/Technology | Version |
|-----------|-------------------|---------|
| Computer Vision | MediaPipe Hands | latest |
| Video Capture | OpenCV (cv2) | latest |
| ML Classification | scikit-learn (KNN) | latest |
| GUI | PySide6 (Qt Widgets) | latest |
| TTS | pyttsx3 | latest |
| Serialization | NumPy, joblib, json | - |
| Python | CPython | 3.10-3.13 |

### 5. User Workflow

1. **Record Gesture**:
   - Run `python cv/record_gestures.py --label zoom --num-samples 30 --frames 30`
   - Press `s` to start recording
   - Perform gesture 30 times
   - Press `n` to save sample
   - Repeat 30 times

2. **Train Model**:
   - Run `python cv/train_classifier.py --data-root data/gestures`
   - Model saved to `models/knn.pkl`

3. **Inference**:
   - Run `python cv/realtime_infer.py --tts`
   - Perform gesture in front of camera → system vocalizes name

4. **GUI Mode**:
   - Run `python -m app.gui_main`
   - Click "Record gesture" → dialog opens → subprocess launches
   - Click "Start inference" → inference subprocess launches

### 6. Key Shortcomings for Redesign

1. **Architectural**:
   - Lack of modularity: CV, GUI, ML - disconnected scripts
   - No service architecture (all CLI)
   - Subprocess approach instead of integration

2. **Functional**:
   - No command system (gesture → action)
   - No database for command/gesture management
   - No voice assistant (STT absent)
   - No cross-platform command executor

3. **UI/UX**:
   - Primitive Qt Widgets interface
   - No Material Design 3
   - No minimalist panel / avatar
   - No visual feedback during recognition

4. **Data**:
   - File-based storage (NPY, JSON) - no queries, statistics
   - No gesture usage history
   - No user settings

