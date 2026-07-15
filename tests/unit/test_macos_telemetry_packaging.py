from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


def _launcher_module():
    path = Path("packaging/macos/gesturebind_launcher.py").resolve()
    spec = importlib.util.spec_from_file_location("gesturebind_launcher_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_launcher_applies_embedded_telemetry_destination(monkeypatch, tmp_path):
    monkeypatch.delenv("DPLM_TELEMETRY_ENDPOINT", raising=False)
    monkeypatch.delenv("DPLM_TELEMETRY_PROJECT_KEY", raising=False)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "release_telemetry.json").write_text(
        json.dumps(
            {
                "endpoint": "https://telemetry.example.test/v1/telemetry/daily",
                "project_key": "public-key",
            }
        ),
        encoding="utf-8",
    )

    module = _launcher_module()
    module._apply_release_telemetry_config(tmp_path)

    assert (
        module.os.environ["DPLM_TELEMETRY_ENDPOINT"]
        == "https://telemetry.example.test/v1/telemetry/daily"
    )
    assert module.os.environ["DPLM_TELEMETRY_PROJECT_KEY"] == "public-key"


def test_release_launcher_rejects_running_directly_from_dmg():
    module = _launcher_module()

    assert module._is_running_from_disk_image(
        Path("/Volumes/GestureBind v0.8.1-beta.4/GestureBind.app/Contents/MacOS/GestureBind")
    )
    assert not module._is_running_from_disk_image(
        Path("/Applications/GestureBind.app/Contents/MacOS/GestureBind")
    )


def test_release_launcher_uses_bundled_branded_flet_client(monkeypatch, tmp_path):
    module = _launcher_module()
    view_dir = tmp_path / "gesturebind_flet"
    (view_dir / "GestureBindUI.app").mkdir(parents=True)
    monkeypatch.delenv("FLET_VIEW_PATH", raising=False)

    module._configure_bundled_flet_view(tmp_path)

    assert module.os.environ["FLET_VIEW_PATH"] == str(view_dir)


def test_release_bundle_does_not_embed_local_mistral_secret():
    build_script = Path("packaging/macos/build_app.sh").read_text(encoding="utf-8")

    assert '--add-data ".env:."' not in build_script
    assert '--add-data ".env.example:."' in build_script


def test_release_dependencies_include_matching_flet_runtime_extras():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert "flet[desktop,web]>=0.85.3" in requirements
    assert "\nflet-desktop" not in requirements


def test_release_build_uses_drag_and_drop_dmg_layout():
    build_script = Path("packaging/macos/build_app.sh").read_text(encoding="utf-8")

    assert "create-dmg" in build_script
    assert "--app-drop-link" in build_script
    assert "--background" in build_script
    assert 'render_app_icon.py "$APP_ICON_PATH"' in build_script
    assert '--icon "$APP_ICON_PATH"' in build_script
    assert '"$ARTIFACT_DIR/INSTALL.txt"' in build_script
    assert '--icon "INSTALL.txt"' in build_script
    assert '--icon "USER_GUIDE.md"' in build_script
    assert '--icon "README.md"' in build_script
    assert "Add :LSUIElement bool true" in build_script
    assert 'FLET_VIEW_DIR="${APP_PATH}/Contents/Frameworks/gesturebind_flet"' in (
        build_script
    )
    assert "Set :CFBundleIdentifier ai.gesturebind.desktop.ui" in build_script
    assert 'rm -f "$FLET_ARCHIVE_PATH"' in build_script


def test_release_build_maps_beta_version_to_macos_bundle_metadata():
    build_script = Path("packaging/macos/build_app.sh").read_text(encoding="utf-8")

    assert 'BUNDLE_SHORT_VERSION="${BUNDLE_VERSION%%-*}"' in build_script
    assert 'BUNDLE_BUILD_VERSION="${BUNDLE_SHORT_VERSION}b${BASH_REMATCH[1]}"' in (
        build_script
    )
    assert "Set :CFBundleShortVersionString ${BUNDLE_SHORT_VERSION}" in build_script
    assert "Set :CFBundleVersion ${BUNDLE_BUILD_VERSION}" in build_script


def test_app_icon_renderer_outputs_valid_icns(tmp_path):
    path = Path("packaging/macos/render_app_icon.py").resolve()
    spec = importlib.util.spec_from_file_location("render_app_icon_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "GestureBind.icns"
    module.render_icns(output)

    master = Image.open(output).convert("RGBA")
    assert master.size == (1024, 1024)
    assert master.getpixel((0, 0))[3] == 0
    assert master.getpixel((512, 512))[3] > 0


def test_dmg_background_has_contrasting_label_area(tmp_path):
    path = Path("packaging/macos/render_dmg_background.py").resolve()
    spec = importlib.util.spec_from_file_location("render_dmg_background_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "background.png"

    module.render(output)

    image = Image.open(output).convert("RGB")
    assert image.size == (840, 560)
    assert sum(image.getpixel((20, 120))) < sum(image.getpixel((20, 420)))
