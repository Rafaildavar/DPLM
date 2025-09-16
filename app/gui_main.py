import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QMessageBox


# ----------------------------------------------
# PySide6 GUI каркас: превью камеры + базовые кнопки
# Комментарии на русском
# ----------------------------------------------

MACOS_BACKEND = cv2.CAP_AVFOUNDATION  # стабильный для macOS (M1)


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
        self.setWindowTitle("DPLM — Gesture Assistant (GUI)")

        self.camera_widget = CameraWidget(self)

        self.btn_infer = QPushButton("Старт инференса (заглушка)")
        self.btn_record = QPushButton("Запись жеста (заглушка)")

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

    def on_infer_clicked(self):
        QMessageBox.information(self, "Инференс", "Здесь будет запуск realtime_infer.py с параметрами")

    def on_record_clicked(self):
        QMessageBox.information(self, "Запись", "Здесь будет запуск record_gestures.py с параметрами")

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
