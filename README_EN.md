# GestureBind: Gesture & Voice Assistant

> **Status: In Active Development (Full Redesign)**  
> Current branch: `redesign`  
> Old version: tag `v0.6.0-old`

## Project Description

Intelligent computer control system using gestures and voice commands. Key feature: the project ships with a recorded base gesture library, while users can add their own gestures and bind them to any commands (opening applications, system actions, custom scripts).

### Main Features

1. **Built-in Base Gesture Library**
   - Pre-recorded static and dynamic gestures for a fast first run
   - Negative classes to protect against accidental triggers
   - Immediate testing of the "gesture -> OS command" scenario

2. **Custom Gesture Training**
   - Intuitive interface for recording gesture samples
   - Automatic machine learning model training
   - Real-time recognition accuracy verification

3. **Command System**
   - Ready-made templates: system, media, custom
   - Bind gestures to any actions
   - Cross-platform execution (Windows + macOS)

4. **Voice Assistant**
   - Speech recognition (Speech-to-Text)
   - Confirmation and hint vocalization (Text-to-Speech)
   - Contextual dialogues and assistance

5. **Modern Interface**
   - Minimalist panel (transparency, drag&drop)
   - Animated avatar assistant
   - Material Design 3 (QML + Qt Quick)
   - Visual feedback during recognition

## Architecture

```
GestureBind/
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
| **Speech-to-Text** | Temporarily disabled |
| **Text-to-Speech** | Temporarily disabled |
| **Command Execution** | subprocess, pyautogui |
| **Platforms** | Windows 10/11, macOS 11+ |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/GestureBind.git
cd GestureBind

# Create and activate virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Known issue on macOS 26 (Tahoe)

On macOS 26 the `PySide6 6.10+` wheels (including 6.11) are blocked by the new
Gatekeeper provenance check (`com.apple.provenance`); Qt silently rejects all
platform plugins from `PySide6/Qt/plugins/platforms` and the app dies with:

```
qt.qpa.plugin: Could not find the Qt platform plugin "cocoa"
This application failed to start because no Qt platform plugin could be initialized.
```

`requirements.txt` pins `PySide6==6.9.3`, which predates the policy change and
works out of the box on macOS 26 + Python 3.13. If you installed a newer
version manually, roll back:

```bash
.venv/bin/pip install --force-reinstall 'PySide6==6.9.3' 'PySide6-Addons==6.9.3' 'PySide6-Essentials==6.9.3' 'shiboken6==6.9.3'
```

### Running

```bash
# 1) Start PostgreSQL
docker compose up -d db

# 2) Run GUI (Flet is the primary version)
PYTHONDONTWRITEBYTECODE=1 python -B -m app.flet_app.main
```

The voice assistant is temporarily disabled: its behavior is not wired into
the current Flet UI yet, so the Assistant tab is hidden instead of exposing
non-working controls. The supported flow is gesture training, gesture list,
gesture-command bindings, and settings.

If the macOS window gets stuck on `Working...`, remove old bytecode caches and
start again:

```bash
find app -name __pycache__ -type d -prune -exec rm -rf {} +
PYTHONDONTWRITEBYTECODE=1 python -B -m app.flet_app.main
```

### Legacy Run

```bash
# Run legacy QML/PySide6 (only if PySide6 is installed)
python -m app.main

# Run old version (for comparison)
git checkout v0.6.0-old
python -m app.gui_main
```

## Workflow

1. **Create Command**
   - Open commands panel
   - Click "Add command"
   - Specify name, platform, action (application/script)

2. **Choose or Train Gesture**
   - Use a base gesture from the built-in library
   - Or record your own gesture 20-30 times in front of camera
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

- Python 3.12 (recommended for the current Flet version)
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
