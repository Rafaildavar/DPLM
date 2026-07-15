#!/usr/bin/env python3
"""Render the GestureBind macOS iconset from the in-app gesture mark."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


CANVAS_SIZE = 1024
GESTURE_GLYPH = "\ue2d4"


def _material_icons_font() -> Path:
    import flet_web

    path = (
        Path(flet_web.__file__).resolve().parent
        / "web"
        / "assets"
        / "fonts"
        / "MaterialIcons-Regular.otf"
    )
    if not path.exists():
        raise FileNotFoundError(f"Material Icons font not found: {path}")
    return path


def _vertical_gradient(top: str, bottom: str) -> Image.Image:
    top_rgb = Image.new("RGB", (1, 1), top).getpixel((0, 0))
    bottom_rgb = Image.new("RGB", (1, 1), bottom).getpixel((0, 0))
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE))
    pixels = image.load()
    for y in range(CANVAS_SIZE):
        amount = y / float(CANVAS_SIZE - 1)
        color = tuple(
            round(start + (end - start) * amount)
            for start, end in zip(top_rgb, bottom_rgb)
        )
        for x in range(CANVAS_SIZE):
            pixels[x, y] = color
    return image


def render_master_icon() -> Image.Image:
    image = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))

    shadow = Image.new("L", image.size, 0)
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle((94, 112, 930, 948), radius=196, fill=210)
    shadow = shadow.filter(ImageFilter.GaussianBlur(34))
    shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_layer.putalpha(shadow)
    image.alpha_composite(shadow_layer)

    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((96, 82, 928, 914), radius=190, fill=255)

    face = _vertical_gradient("#1D343A", "#12191D").convert("RGBA")
    face.putalpha(mask)
    image.alpha_composite(face)

    border = ImageDraw.Draw(image)
    border.rounded_rectangle(
        (98, 84, 926, 912),
        radius=188,
        outline="#5DD7E3",
        width=10,
    )

    font = ImageFont.truetype(str(_material_icons_font()), size=570)
    glyph_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glyph_draw = ImageDraw.Draw(glyph_layer)
    box = glyph_draw.textbbox((0, 0), GESTURE_GLYPH, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    glyph_draw.text(
        ((CANVAS_SIZE - width) / 2 - box[0], 480 - height / 2 - box[1]),
        GESTURE_GLYPH,
        font=font,
        fill="#58D2DF",
    )

    glow = glyph_layer.getchannel("A").filter(ImageFilter.GaussianBlur(22))
    glow_layer = Image.new("RGBA", image.size, (59, 211, 224, 0))
    glow_layer.putalpha(glow.point(lambda value: round(value * 0.28)))
    image.alpha_composite(glow_layer)
    image.alpha_composite(glyph_layer)

    return image


def render_icns(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    render_master_icon().save(output, format="ICNS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render_icns(args.output)


if __name__ == "__main__":
    main()
