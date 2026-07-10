from scripts.check_repository_hygiene import (
    FORBIDDEN_TRACKED_PATHS,
    RELEASE_MODEL_FILES,
    REQUIRED_RELEASE_FILES,
    tracked_files,
    validate_release_metadata,
    validate_tracked_files,
)


def _minimal_release_tree() -> set[str]:
    return set(REQUIRED_RELEASE_FILES) | set(RELEASE_MODEL_FILES)


def test_minimal_release_tree_is_valid() -> None:
    assert validate_tracked_files(_minimal_release_tree()) == []


def test_generated_and_legacy_paths_are_rejected() -> None:
    paths = _minimal_release_tree() | {
        ".env",
        ".venv/bin/python",
        "app/qml/MainWindow.qml",
        "data/gestures/SwipeLeft/sample_001.npy",
        "models/experiments/candidate.pkl",
        "outputs/report.json",
    }

    errors = validate_tracked_files(paths)

    assert any("forbidden tracked file: .env" in error for error in errors)
    assert any("app/qml/MainWindow.qml" in error for error in errors)
    assert any("data/gestures" in error for error in errors)
    assert any("unexpected release model artifact" in error for error in errors)
    assert any("outputs/report.json" in error for error in errors)


def test_missing_release_asset_is_rejected() -> None:
    paths = _minimal_release_tree() - {"models/knn.pkl"}

    errors = validate_tracked_files(paths)

    assert "required release file is missing: models/knn.pkl" in errors


def test_forbidden_file_contract_contains_secrets() -> None:
    assert ".env" in FORBIDDEN_TRACKED_PATHS


def test_release_metadata_versions_must_match(tmp_path) -> None:
    (tmp_path / "VERSION").write_text("1.2.3\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("MVP `1.2.3`", encoding="utf-8")
    (tmp_path / "RELEASE_NOTES.md").write_text(
        "# GestureBind v1.2.3", encoding="utf-8"
    )
    (tmp_path / "CHANGELOG.md").write_text("## v1.2.3", encoding="utf-8")

    assert validate_release_metadata(tmp_path) == []

    (tmp_path / "RELEASE_NOTES.md").write_text(
        "# GestureBind v1.2.2", encoding="utf-8"
    )
    assert "release version mismatch" in validate_release_metadata(tmp_path)[0]


def test_current_git_index_matches_release_contract() -> None:
    assert validate_tracked_files(tracked_files()) == []
    assert validate_release_metadata() == []
