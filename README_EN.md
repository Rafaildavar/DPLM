# DPLM: Gesture & Voice Assistant

> **Status: In Active Development (Full Redesign)**  
> Current branch: `redesign`  
> Old version: tag `v0.6.0-old`

## Project Description

Intelligent computer control system using gestures and voice commands. Key feature: users train the system with their own gestures and bind them to any commands (opening applications, system actions, custom scripts).

### Main Features

1. **Custom Gesture Training**
   - Intuitive interface for recording gesture samples
   - Automatic machine learning model training
   - Real-time recognition accuracy verification

2. **Command System**
   - Ready-made templates: system, media, custom
   - Bind gestures to any actions
   - Cross-platform execution (Windows + macOS)

3. **Voice Assistant**
   - Speech recognition (Speech-to-Text)
   - Confirmation and hint vocalization (Text-to-Speech)
   - Contextual dialogues and assistance

4. **Modern Interface**
   - Minimalist panel (transparency, drag&drop)
   - Animated avatar assistant
   - Material Design 3 (QML + Qt Quick)
   - Visual feedback during recognition

## Implemented So Far (Current State)

- ✅ Added **resilient startup** via `python -m app.start` (dependency preflight with actionable diagnostics instead of hard crash).
- ✅ CV pipeline for gesture recording/training/realtime inference is available (`cv/record_gestures.py`, `cv/train_classifier.py`, `cv/realtime_infer.py`).
- ✅ Feature improvements are integrated: hand geometry, deterministic Left/Right ordering, hand-presence masks, EMA smoothing, and stable-window voting.
- ✅ Training now uses `StandardScaler + KNN(weights="distance")` and stores baseline metrics in `models/train_metrics.json`.
- ⚠️ Full GUI (`app.main`) requires full dependency stack (`PySide6`, CV/ML packages, DB packages).

## Architecture

```
DPLM/
├── app/
│   ├── qml/                # QML interfaces (Material Design 3)
│   ├── backend/            # Business logic (commands, ML)
│   ├── models/             # Data models (PostgreSQL + SQLAlchemy)
│   ├── services/           # Services (CV, TTS, STT, executor)
│   ├── resources/          # Resources (icons, avatar, sounds)
│   └── main.py             # Entry point (Qt QML Application)
├── data/
│   └── gestures/           # Gesture samples (NPY)
├── models/
│   └── user_gestures/      # User-trained models
├── docs/                   # Documentation (RU + EN)
└── requirements.txt        # Python dependencies
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **UI Framework** | PySide6 (QML + Qt Quick) |
| **UI Design** | Material Design 3 |
| **Computer Vision** | MediaPipe Hands (baseline), YOLO11 Pose (optional) |
| **ML Classification** | scikit-learn (KNN/SVM), LSTM/GRU (complex gestures) |
| **Database** | PostgreSQL Embedded (pg_embed) |
| **ORM** | SQLAlchemy + Alembic (migrations) |
| **Speech-to-Text** | SpeechRecognition, Vosk (offline) |
| **Text-to-Speech** | pyttsx3 → Coqui TTS |
| **Command Execution** | subprocess, pyautogui |
| **Platforms** | Windows 10/11, macOS 11+ |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/DPLM.git
cd DPLM

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Recommended startup (resilient bootstrap + environment preflight)
python -m app.start

# Direct GUI startup (only when all dependencies are installed)
python -m app.main

# Shell launcher
bash scripts/launch_app.sh
```

If dependencies are missing, `app.start` will not crash; it prints install guidance and runs environment diagnostics.

## Workflow

1. **Create Command**
   - Open commands panel
   - Click "Add command"
   - Specify name, platform, action (application/script)

2. **Train Gesture**
   - Select command → "Assign gesture"
   - Perform gesture 20-30 times in front of camera
   - System automatically trains model
   - Verify accuracy in preview mode

3. **Usage**
   - System runs in background (minimalist panel)
   - Perform gesture → command executes
   - Avatar provides visual feedback
   - Voice hints when needed

## Documentation

- [Old Version Architecture](docs/ARCHITECTURE_OLD.md) (RU + EN)
- [User Guide](docs/USER_GUIDE.md) (in development)
- [Developer Guide](docs/DEVELOPER.md) (in development)
- [Commands List](docs/COMMANDS.md) (in development)
- [CV Models Comparison](docs/CV_COMPARISON.md) (in development)
- [Testing](docs/TESTING.md) (in development)

## Health Check

```bash
# Environment diagnostics
python scripts/check_env.py

# Bootstrap smoke run
python -m app.start

# Startup-layer unit tests
pytest -q -o addopts='' tests/unit/test_start_launcher.py
```

## Development

### Branch Structure

- `main` — stable version (releases)
- `dev` — development (feature integration)
- `redesign` — **current branch** (full redesign)
- `v0.6.0-old` — old version tag

### Commits

Message format:
```
<type>: brief description

Detailed description (optional)

<type>: feat, fix, docs, refactor, test, chore
```

Example:
```
feat: implement gesture training service with LSTM support

- Add app/services/gesture_trainer.py
- Support both KNN and LSTM classifiers
- Integrate with QML UI via Qt signals
```

## Requirements

- Python 3.10+
- macOS 11+ (Apple Silicon / Intel) or Windows 10/11
- Webcam (1280×720 or higher)
- 4GB RAM minimum (8GB recommended)
- 500MB free disk space

## License

MIT License (see LICENSE)

## Author

Developed as part of GUAP diploma project (2025)

## Contact

- GitHub Issues: [issues and suggestions]
- Email: rafaildavar@gmail.com

---

**Note**: Project is in active development. Old version available at tag `v0.6.0-old` for reference to original implementation.

