#!/usr/bin/env python3
"""Render the Finder background used by the macOS drag-and-drop DMG."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 840
HEIGHT = 560
BACKGROUND = "#17191D"
FOREGROUND = "#F4F6F8"
MUTED = "#AAB0B7"
ACCENT = "#50C8D8"
GUIDE_BACKGROUND = "#E8EEF1"
GUIDE_FOREGROUND = "#20262B"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSRounded.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    draw.text(((WIDTH - width) / 2, y), text, font=font, fill=fill)


def render(output: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 292, WIDTH, HEIGHT), fill=GUIDE_BACKGROUND)

    _centered_text(
        draw,
        42,
        "Установите GestureBind",
        font=_font(32, bold=True),
        fill=FOREGROUND,
    )
    _centered_text(
        draw,
        88,
        "Перетащите приложение в папку Applications",
        font=_font(17),
        fill=MUTED,
    )

    arrow_y = 226
    draw.line((322, arrow_y, 518, arrow_y), fill=ACCENT, width=5)
    draw.polygon(
        ((518, arrow_y), (498, arrow_y - 13), (498, arrow_y + 13)),
        fill=ACCENT,
    )

    _centered_text(
        draw,
        326,
        "Документация",
        font=_font(18, bold=True),
        fill=GUIDE_FOREGROUND,
    )
    _centered_text(
        draw,
        528,
        "Сначала прочитайте INSTALL.txt",
        font=_font(13),
        fill="#59636A",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
