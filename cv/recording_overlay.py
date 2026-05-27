"""Polished camera overlay for the gesture sample recording workflow."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_BOLD_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = _BOLD_FONT_CANDIDATES if bold else _FONT_CANDIDATES
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _panel(
    frame_bgr: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    alpha: float = 0.78,
) -> None:
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (18, 22, 29), -1)
    cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0, dst=frame_bgr)


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    *,
    size: int,
    color: tuple[int, int, int],
    bold: bool = False,
) -> None:
    draw.text(xy, value, font=_font(size, bold), fill=color)


def _hint(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    key: str,
    label: str,
) -> int:
    white = (244, 246, 249)
    muted = (186, 193, 203)
    draw.rounded_rectangle((x, y, x + 34, y + 32), radius=7, fill=(56, 64, 75))
    _text(draw, (x + 11, y + 4), key, size=18, color=white, bold=True)
    _text(draw, (x + 45, y + 6), label, size=17, color=muted)
    text_box = draw.textbbox((0, 0), label, font=_font(17))
    return x + 45 + (text_box[2] - text_box[0]) + 34


def draw_recording_overlay(
    frame_bgr: np.ndarray,
    *,
    label: str,
    saved: int,
    target_samples: int,
    recording: bool,
    frame_count: int,
    sequence_length: int,
    hands_count: int,
) -> np.ndarray:
    """Render user-facing status panels after detection has processed the frame."""
    height, width = frame_bgr.shape[:2]
    white = (244, 246, 249)
    muted = (186, 193, 203)
    green = (74, 196, 111)
    red = (244, 79, 88)
    amber = (241, 180, 60)
    margin = 24

    top_height = min(116, max(94, height // 6))
    bottom_height = min(126, max(108, height // 5))
    _panel(frame_bgr, (0, 0), (width, top_height))
    _panel(frame_bgr, (0, height - bottom_height), (width, height))

    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    display_label = label if len(label) <= 34 else f"{label[:31]}..."
    _text(draw, (margin, 18), "Запись жеста", size=28, color=white, bold=True)
    _text(draw, (margin, 57), f"Жест: {display_label}", size=18, color=muted)

    sample_text = f"Пример {min(saved + 1, target_samples)} из {target_samples}"
    sample_box = draw.textbbox((0, 0), sample_text, font=_font(18, True))
    _text(
        draw,
        (width - margin - (sample_box[2] - sample_box[0]), 27),
        sample_text,
        size=18,
        color=white,
        bold=True,
    )

    bar_y = top_height - 16
    draw.rounded_rectangle(
        (margin, bar_y, width - margin, bar_y + 6),
        radius=3,
        fill=(67, 73, 83),
    )
    sample_progress = min(1.0, saved / max(1, target_samples))
    if sample_progress:
        draw.rounded_rectangle(
            (margin, bar_y, margin + int((width - 2 * margin) * sample_progress), bar_y + 6),
            radius=3,
            fill=green,
        )

    if recording and frame_count >= sequence_length:
        status = "Пример готов. Нажмите N для сохранения"
        status_color = green
    elif recording:
        status = f"Идет запись: {min(frame_count, sequence_length)} из {sequence_length} кадров"
        status_color = red
    elif hands_count:
        status = "Рука найдена. Нажмите S для записи"
        status_color = green
    else:
        status = "Покажите руку в кадре"
        status_color = amber

    status_y = height - bottom_height + 20
    draw.ellipse((margin, status_y + 9, margin + 14, status_y + 23), fill=status_color)
    _text(draw, (margin + 28, status_y), status, size=22, color=white, bold=True)

    hints_y = height - 50
    next_x = _hint(draw, margin, hints_y, "S", "Начать / повторить")
    next_x = _hint(draw, next_x, hints_y, "N", "Сохранить")
    _hint(draw, next_x, hints_y, "Q", "Завершить")

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
