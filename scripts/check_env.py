import importlib
import sys
from typing import List, Tuple


# ---------------------------------------------
# Проверка окружения: какие библиотеки установлены
# Запуск: python scripts/check_env.py
# ---------------------------------------------

TO_CHECK: List[Tuple[str, str]] = [
    ("numpy", "__version__"),
    ("cv2", "__version__"),
    ("mediapipe", "__version__"),
    ("sklearn", "__version__"),
    ("joblib", "__version__"),
    ("torch", "__version__"),
    ("flet", "__version__"),
    ("sqlalchemy", "__version__"),
    ("mlflow", "__version__"),
    ("pyttsx3", "__version__"),
    ("speech_recognition", "__version__"),  # SpeechRecognition
    ("vosk", "__version__"),
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print("Checking installed packages:\n")
    missing = []
    for mod, attr in TO_CHECK:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, attr, "?")
            print(f"[OK] {mod}: {ver}")
        except Exception as e:
            print(f"[MISS] {mod}: not installed ({e.__class__.__name__}: {e})")
            missing.append(mod)

    if missing:
        print("\nTo install missing (example):")
        replacements = {
            "speech_recognition": "SpeechRecognition",
        }
        pkgs = [replacements.get(m, m) for m in missing]
        print("pip install " + " ".join(pkgs))
    else:
        print("\nAll required packages are installed.")


if __name__ == "__main__":
    main()

