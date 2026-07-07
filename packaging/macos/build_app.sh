#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

APP_NAME="${APP_NAME:-GestureBind}"
VERSION="${GESTUREBIND_VERSION:-${GESTUREFLOW_VERSION:-local}}"
ARTIFACT_DIR="outputs/release/${APP_NAME}-macos-${VERSION}"
ZIP_PATH="outputs/release/${APP_NAME}-macos-${VERSION}.zip"

python -m pip install --upgrade pip setuptools wheel
python -m pip install pyinstaller

python - <<'PY'
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import flet_desktop
import flet_desktop.version
from flet_desktop import get_artifact_filename, get_package_bin_dir

artifact = get_artifact_filename()
target = Path(get_package_bin_dir()) / artifact
url = os.environ.get(
    "FLET_CLIENT_URL",
    f"https://github.com/flet-dev/flet/releases/download/"
    f"v{flet_desktop.version.version}/{artifact}",
)
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    print(f"[macos-bundle] Downloading Flet desktop client: {url}")
    urllib.request.urlretrieve(url, target)
else:
    print(f"[macos-bundle] Flet desktop client already bundled: {target}")
print(f"[macos-bundle] Client artifact: {target}")
PY

python -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --osx-bundle-identifier "ai.gesturebind.desktop" \
  --collect-all flet \
  --collect-all flet_desktop \
  --collect-all mediapipe \
  --collect-all cv2 \
  --collect-submodules sklearn \
  --collect-submodules skops \
  --collect-submodules joblib \
  --collect-submodules torch \
  --hidden-import cv2 \
  --hidden-import mediapipe.tasks.python.vision \
  --add-data "models:models" \
  --add-data "configs:configs" \
  --add-data "alembic:alembic" \
  --add-data ".env.example:." \
  packaging/macos/gesturebind_launcher.py

APP_PATH="dist/${APP_NAME}.app"
PLIST_PATH="${APP_PATH}/Contents/Info.plist"

if [[ -f "$PLIST_PATH" ]]; then
  /usr/libexec/PlistBuddy -c \
    "Add :NSCameraUsageDescription string GestureBind needs camera access to recognize hand gestures." \
    "$PLIST_PATH" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c \
    "Set :NSCameraUsageDescription GestureBind needs camera access to recognize hand gestures." \
    "$PLIST_PATH"

  /usr/libexec/PlistBuddy -c \
    "Add :NSMicrophoneUsageDescription string GestureBind can use microphone access for voice assistant features." \
    "$PLIST_PATH" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c \
    "Set :NSMicrophoneUsageDescription GestureBind can use microphone access for voice assistant features." \
    "$PLIST_PATH"

  /usr/libexec/PlistBuddy -c \
    "Add :NSAppleEventsUsageDescription string GestureBind can execute user-configured macOS automation commands." \
    "$PLIST_PATH" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c \
    "Set :NSAppleEventsUsageDescription GestureBind can execute user-configured macOS automation commands." \
    "$PLIST_PATH"
fi

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$APP_PATH"
fi

rm -rf "$ARTIFACT_DIR"
mkdir -p "$ARTIFACT_DIR"
cp -R "$APP_PATH" "$ARTIFACT_DIR/"
cp packaging/macos/RUN_MACOS.md "$ARTIFACT_DIR/"
cp README.md "$ARTIFACT_DIR/README.md"

rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$ARTIFACT_DIR" "$ZIP_PATH"
echo "[macos-bundle] Artifact: $ZIP_PATH"
