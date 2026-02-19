from unittest.mock import patch

from app import start


def test_missing_modules_detected():
    with patch("importlib.import_module", side_effect=ImportError("x")):
        missing = start._missing_modules(["a", "b"])
    assert missing == ["a", "b"]


def test_main_returns_zero_when_dependencies_missing():
    with patch("app.start._missing_modules", return_value=["PySide6"]):
        with patch("app.start._run_env_check"):
            rc = start.main()
    assert rc == 0
