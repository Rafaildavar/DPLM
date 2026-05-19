"""
Конфигурация pytest и общие фикстуры
pytest configuration and shared fixtures
"""
import sys
from pathlib import Path
import pytest

try:
    from PySide6.QtWidgets import QApplication  # noqa: F401
except ImportError:  # Flet-only test runs do not require Qt.
    QApplication = None  # type: ignore[assignment]

# Добавить корневую директорию проекта в PYTHONPATH
# Add project root directory to PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Используем встроенную фикстуру qapp из pytest-qt
# Use built-in qapp fixture from pytest-qt
# (автоматически управляет QApplication lifecycle)
# (automatically manages QApplication lifecycle)


@pytest.fixture
def test_data_dir():
    """
    Фикстура для пути к тестовым данным
    Fixture for test data directory path
    """
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture
def temp_db(tmp_path):
    """
    Фикстура для временной БД (SQLite) для тестов
    Fixture for temporary database (SQLite) for tests
    
    Args:
        tmp_path: встроенная pytest фикстура для временной директории
                  built-in pytest fixture for temporary directory
    
    Returns:
        Path: путь к временной БД / path to temporary database
    """
    db_path = tmp_path / "test.db"
    return db_path


@pytest.fixture
def mock_camera():
    """
    Фикстура для мок-объекта камеры (избегаем реального видео в тестах)
    Fixture for mock camera object (avoid real video in tests)
    """
    class MockCamera:
        def __init__(self):
            self.is_opened = True
            self.frame_count = 0
        
        def isOpened(self):
            return self.is_opened
        
        def read(self):
            import numpy as np
            # Возвращаем фейковый кадр 640x480
            # Return fake frame 640x480
            self.frame_count += 1
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return True, fake_frame
        
        def release(self):
            self.is_opened = False
    
    return MockCamera()


@pytest.fixture
def sample_gesture_data():
    """
    Фикстура с примером данных жеста (нормализованные ландмарки)
    Fixture with sample gesture data (normalized landmarks)
    
    Returns:
        numpy.ndarray: массив формы (seq_len, 42) для одной руки
                       array of shape (seq_len, 42) for one hand
    """
    import numpy as np
    seq_len = 30
    num_landmarks = 21
    coords_per_landmark = 2
    feature_dim = num_landmarks * coords_per_landmark  # 42
    
    # Генерируем случайные нормализованные координаты
    # Generate random normalized coordinates
    data = np.random.rand(seq_len, feature_dim).astype(np.float32)
    return data
