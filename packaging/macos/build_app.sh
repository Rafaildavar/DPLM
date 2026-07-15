#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

APP_NAME="${APP_NAME:-GestureBind}"
VERSION="${GESTUREBIND_VERSION:-${GESTUREFLOW_VERSION:-local}}"
ARTIFACT_DIR="outputs/release/${APP_NAME}-macos-${VERSION}"
ZIP_PATH="outputs/release/${APP_NAME}-macos-${VERSION}.zip"
DMG_PATH="outputs/release/${APP_NAME}-macos-${VERSION}.dmg"
DMG_BACKGROUND_PATH="outputs/release/.${APP_NAME}-dmg-background.png"
APP_ICON_PATH="outputs/release/.${APP_NAME}.icns"
TELEMETRY_CONFIG_PATH="configs/release_telemetry.json"
GENERATED_TELEMETRY_CONFIG=0

cleanup() {
  rm -f "$DMG_BACKGROUND_PATH"
  rm -f "$APP_ICON_PATH"
  if [[ "$GENERATED_TELEMETRY_CONFIG" == "1" ]]; then
    rm -f "$TELEMETRY_CONFIG_PATH"
  fi
}
trap cleanup EXIT

if [[ -n "${GESTUREBIND_TELEMETRY_ENDPOINT:-}" ]]; then
  python - <<'PY'
import json
import os
from pathlib import Path

Path("configs/release_telemetry.json").write_text(
    json.dumps(
        {
            "endpoint": os.environ["GESTUREBIND_TELEMETRY_ENDPOINT"].strip(),
            "project_key": os.environ.get(
                "GESTUREBIND_TELEMETRY_PROJECT_KEY", ""
            ).strip(),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
  GENERATED_TELEMETRY_CONFIG=1
fi

python -m pip install --upgrade pip "setuptools<82" wheel
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

mkdir -p outputs/release
python packaging/macos/render_app_icon.py "$APP_ICON_PATH"

python -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --icon "$APP_ICON_PATH" \
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
  --add-data "VERSION:." \
  packaging/macos/gesturebind_launcher.py

APP_PATH="dist/${APP_NAME}.app"
PLIST_PATH="${APP_PATH}/Contents/Info.plist"
BUNDLE_VERSION="${VERSION#v}"
BUNDLE_SHORT_VERSION="${BUNDLE_VERSION%%-*}"
BUNDLE_BUILD_VERSION="$BUNDLE_SHORT_VERSION"
FLET_VIEW_DIR="${APP_PATH}/Contents/Frameworks/gesturebind_flet"
FLET_CLIENT_APP="${FLET_VIEW_DIR}/GestureBindUI.app"

if [[ "$BUNDLE_VERSION" =~ -beta\.([0-9]+)$ ]]; then
  BUNDLE_BUILD_VERSION="${BUNDLE_SHORT_VERSION}b${BASH_REMATCH[1]}"
elif [[ "$BUNDLE_VERSION" =~ -alpha\.([0-9]+)$ ]]; then
  BUNDLE_BUILD_VERSION="${BUNDLE_SHORT_VERSION}a${BASH_REMATCH[1]}"
elif [[ "$BUNDLE_VERSION" =~ -rc\.([0-9]+)$ ]]; then
  BUNDLE_BUILD_VERSION="${BUNDLE_SHORT_VERSION}fc${BASH_REMATCH[1]}"
fi

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

  /usr/libexec/PlistBuddy -c \
    "Add :LSUIElement bool true" \
    "$PLIST_PATH" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c \
    "Set :LSUIElement true" \
    "$PLIST_PATH"

  if [[ "$BUNDLE_SHORT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    /usr/libexec/PlistBuddy -c \
      "Add :CFBundleShortVersionString string ${BUNDLE_SHORT_VERSION}" \
      "$PLIST_PATH" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c \
      "Set :CFBundleShortVersionString ${BUNDLE_SHORT_VERSION}" \
      "$PLIST_PATH"

    /usr/libexec/PlistBuddy -c \
      "Add :CFBundleVersion string ${BUNDLE_BUILD_VERSION}" \
      "$PLIST_PATH" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c \
      "Set :CFBundleVersion ${BUNDLE_BUILD_VERSION}" \
      "$PLIST_PATH"
  fi
fi

FLET_ARCHIVE_PATH="$(
  find "$APP_PATH/Contents" \
    -type f \
    -path "*/flet_desktop/app/flet-macos.tar.gz" \
    -print \
    -quit
)"
if [[ -z "$FLET_ARCHIVE_PATH" ]]; then
  echo "[macos-bundle] bundled Flet macOS client archive not found" >&2
  exit 1
fi

rm -rf "$FLET_VIEW_DIR"
mkdir -p "$FLET_VIEW_DIR"
tar -xzf "$FLET_ARCHIVE_PATH" -C "$FLET_VIEW_DIR"
if [[ ! -d "${FLET_VIEW_DIR}/Flet.app" ]]; then
  echo "[macos-bundle] Flet.app not found in desktop client archive" >&2
  exit 1
fi
mv "${FLET_VIEW_DIR}/Flet.app" "$FLET_CLIENT_APP"
rm -f "$FLET_ARCHIVE_PATH"

FLET_PLIST_PATH="${FLET_CLIENT_APP}/Contents/Info.plist"
cp "$APP_ICON_PATH" "${FLET_CLIENT_APP}/Contents/Resources/GestureBind.icns"
/usr/libexec/PlistBuddy -c "Set :CFBundleName GestureBind" "$FLET_PLIST_PATH"
/usr/libexec/PlistBuddy -c \
  "Add :CFBundleDisplayName string GestureBind" \
  "$FLET_PLIST_PATH" 2>/dev/null || \
/usr/libexec/PlistBuddy -c \
  "Set :CFBundleDisplayName GestureBind" \
  "$FLET_PLIST_PATH"
/usr/libexec/PlistBuddy -c \
  "Set :CFBundleIdentifier ai.gesturebind.desktop.ui" \
  "$FLET_PLIST_PATH"
/usr/libexec/PlistBuddy -c \
  "Set :CFBundleIconFile GestureBind.icns" \
  "$FLET_PLIST_PATH"
/usr/libexec/PlistBuddy -c "Delete :CFBundleIconName" "$FLET_PLIST_PATH" \
  2>/dev/null || true
if [[ "$BUNDLE_SHORT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  /usr/libexec/PlistBuddy -c \
    "Set :CFBundleShortVersionString ${BUNDLE_SHORT_VERSION}" \
    "$FLET_PLIST_PATH"
  /usr/libexec/PlistBuddy -c \
    "Set :CFBundleVersion ${BUNDLE_BUILD_VERSION}" \
    "$FLET_PLIST_PATH"
fi

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$FLET_CLIENT_APP"
  codesign --force --deep --sign - "$APP_PATH"
fi

rm -rf "$ARTIFACT_DIR"
mkdir -p "$ARTIFACT_DIR"
cp -R "$APP_PATH" "$ARTIFACT_DIR/"
cp packaging/macos/INSTALL.txt "$ARTIFACT_DIR/INSTALL.txt"
cp docs/USER_GUIDE.md "$ARTIFACT_DIR/USER_GUIDE.md"
cp README.md "$ARTIFACT_DIR/README.md"

rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$ARTIFACT_DIR" "$ZIP_PATH"

if [[ "${GESTUREBIND_LOW_DISK_BUILD:-0}" == "1" ]]; then
  rm -rf build dist
fi

python packaging/macos/render_dmg_background.py "$DMG_BACKGROUND_PATH"

if ! command -v create-dmg >/dev/null 2>&1; then
  echo "[macos-bundle] create-dmg is required (brew install create-dmg)" >&2
  exit 1
fi

rm -f "$DMG_PATH"
create-dmg \
  --volname "${APP_NAME} ${VERSION}" \
  --volicon "$APP_ICON_PATH" \
  --background "$DMG_BACKGROUND_PATH" \
  --window-pos 200 120 \
  --window-size 840 560 \
  --text-size 13 \
  --icon-size 112 \
  --icon "${APP_NAME}.app" 200 226 \
  --hide-extension "${APP_NAME}.app" \
  --app-drop-link 640 226 \
  --icon "INSTALL.txt" 250 430 \
  --icon "USER_GUIDE.md" 420 430 \
  --icon "README.md" 590 430 \
  --format UDZO \
  --no-internet-enable \
  --overwrite \
  "$DMG_PATH" \
  "$ARTIFACT_DIR"

echo "[macos-bundle] ZIP artifact: $ZIP_PATH"
echo "[macos-bundle] DMG artifact: $DMG_PATH"
