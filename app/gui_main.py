import sys
import subprocess
from pathlib import Path
import cv2
import os

import numpy as np
import PySide6

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QDialog,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QCheckBox,
    QDialogButtonBox,
)
import os
import PySide6

# Найдите точный путь к плагинам
plugin_path = os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
print(f"Plugin path: {plugin_path}")  # Для отладки
# ----------------------------------------------
# PySide6 GUI каркас: превью камеры + кнопки и запуск скриптов
# Комментарии на русском
# ----------------------------------------------
import os
os.environ['QT_QPA_PLATFORM'] = 'cocoa'
MACOS_BACKEND = cv2.CAP_AVFOUNDATION  # стабильный для macOS (M1)


class RecordDialog(QDialog):
    """Диалог параметров записи жестов."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Параметры записи жеста")
        form = QFormLayout(self)

        self.label_edit = QLineEdit(self)
        self.label_edit.setPlaceholderText("например, zoom_one")

        self.frames_spin = QSpinBox(self)
        self.frames_spin.setRange(5, 300)
        self.frames_spin.setValue(30)

        self.samples_spin = QSpinBox(self)
        self.samples_spin.setRange(1, 200)
        self.samples_spin.setValue(30)

        self.twohands_chk = QCheckBox("Две руки", self)

        form.addRow("Label:", self.label_edit)
        form.addRow("Frames per sample:", self.frames_spin)
        form.addRow("Num samples:", self.samples_spin)
        form.addRow(self.twohands_chk)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addWidget(btns)

    def get_params(self):
        return {
            "label": self.label_edit.text().strip(),
            "frames": int(self.frames_spin.value()),
            "num_samples": int(self.samples_spin.value()),
            "two_hands": bool(self.twohands_chk.isChecked()),
        }


class CameraWidget(QWidget):
    """Виджет превью камеры с таймером опроса кадров."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("Камера не запущена")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

        # Инициализация OpenCV захвата
        self.cap = cv2.VideoCapture(0, MACOS_BACKEND)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.label.setText("Ошибка доступа к камере. Проверьте разрешения в macOS")

        # Таймер 30 FPS
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def update_frame(self) -> None:
        if not self.cap.isOpened():
            return
        ok, frame_bgr = self.cap.read()
        if not ok:
            return
        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def close(self) -> None:
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        super().close()


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GestureBind — Gesture Assistant (GUI)")

        self.camera_widget = CameraWidget(self)

        self.btn_infer = QPushButton("Старт инференса")
        self.btn_record = QPushButton("Запись жеста")

        self.btn_infer.clicked.connect(self.on_infer_clicked)
        self.btn_record.clicked.connect(self.on_record_clicked)

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_infer)
        buttons.addWidget(self.btn_record)

        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.addWidget(self.camera_widget)
        layout.addLayout(buttons)

        self.setCentralWidget(root)
        self.resize(960, 640)

    def _run_subprocess(self, args: list[str]) -> None:
        """Запуск внешнего процесса неблокирующе."""
        try:
            subprocess.Popen(args)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка запуска", f"Не удалось запустить: {' '.join(args)}\n{e}")

    def on_infer_clicked(self):
        # Автоматически подстроится под feature_dim в realtime_infer.py
        py = sys.executable
        script = str(Path("cv/realtime_infer.py").resolve())
        self._run_subprocess([py, script, "--tts"])  # можно добавить --window, --two-hands

    def on_record_clicked(self):
        dlg = RecordDialog(self)
        if dlg.exec() == QDialog.Accepted:
            p = dlg.get_params()
            if not p["label"]:
                QMessageBox.warning(self, "Внимание", "Label не может быть пустым")
                return
            py = sys.executable
            script = str(Path("../cv/record_gestures.py").resolve())
            args = [
                py,
                script,
                "--label", p["label"],
                "--frames", str(p["frames"]),
                "--num-samples", str(p["num_samples"]),
            ]
            if p["two_hands"]:
                args.append("--two-hands")
            self._run_subprocess(args)

    def closeEvent(self, event):
        try:
            self.camera_widget.close()
        except Exception:
            pass
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
