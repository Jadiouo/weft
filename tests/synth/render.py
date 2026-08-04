"""合成投影片與講者影格的繪製。SDD §5.1（A）。

投影片刻意做出 §1.3 列的素材特性：密集中文、文言文、**直排文字**、版面帶
語意（箭頭、雙欄、色彩編碼）。合成素材若只有橫排標題，測出來的分數對真實
講經影片沒有參考價值。
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SERIF = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
SANS = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


@functools.lru_cache(maxsize=64)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """取 ttc 中的繁體（TC）字面。index 不寫死——不同發行版的排列可能不同。"""
    for index in range(8):
        try:
            font = ImageFont.truetype(path, size, index=index)
        except OSError:
            break
        if "TC" in font.getname()[0]:
            return font
    return ImageFont.truetype(path, size, index=0)


# --------------------------------------------------------------------------
# 投影片版型
# --------------------------------------------------------------------------

BG = (250, 248, 243)
INK = (28, 28, 32)
ACCENT = (86, 42, 120)


def _new_slide(size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", size, BG)
    return img, ImageDraw.Draw(img)


def _title(draw: ImageDraw.ImageDraw, text: str, width: int, scale: float) -> int:
    font = load_font(SANS, int(30 * scale))
    draw.text((int(40 * scale), int(24 * scale)), text, font=font, fill=ACCENT)
    y = int(70 * scale)
    draw.line([(int(40 * scale), y), (width - int(40 * scale), y)], fill=ACCENT, width=max(1, int(2 * scale)))
    return y + int(20 * scale)


def _draw_vertical(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, size: int, fill=INK) -> None:
    """直排文字：逐字往下堆。PIL 沒有直排排版，這正好也是真實投影片的樣子。"""
    font = load_font(SERIF, size)
    step = int(size * 1.15)
    for i, ch in enumerate(text):
        draw.text((x, y + i * step), ch, font=font, fill=fill)


def render_slide(
    layout: str,
    content: dict,
    size: tuple[int, int],
    reveal: int | None = None,
) -> Image.Image:
    """畫一張投影片。

    `reveal` 為逐條動畫用：只顯示前 N 個項目，其餘留白（版面位置不變，
    這正是「內容單調增加」的視覺形式）。
    """
    width, height = size
    scale = height / 720.0
    img, draw = _new_slide(size)
    y = _title(draw, content["title"], width, scale)

    items = content.get("items", [])
    shown = items if reveal is None else items[:reveal]

    if layout == "vertical":
        # 直排經文：由右至左
        x = width - int(70 * scale)
        for line in shown:
            _draw_vertical(draw, line, x, y + int(10 * scale), int(24 * scale))
            x -= int(38 * scale)

    elif layout == "two_column":
        col_w = (width - int(120 * scale)) // 2
        font_a = load_font(SERIF, int(22 * scale))
        font_b = load_font(SANS, int(19 * scale))
        draw.rectangle(
            [int(40 * scale), y, int(40 * scale) + col_w, height - int(30 * scale)],
            fill=(240, 236, 248),
        )
        for i, line in enumerate(shown):
            left, right = (line.split("｜") + [""])[:2]
            yy = y + int(20 * scale) + i * int(46 * scale)
            draw.text((int(52 * scale), yy), left, font=font_a, fill=INK)
            draw.text((int(80 * scale) + col_w, yy), right, font=font_b, fill=(60, 60, 70))

    elif layout == "arrow":
        font = load_font(SANS, int(22 * scale))
        for i, line in enumerate(shown):
            yy = y + int(24 * scale) + i * int(58 * scale)
            draw.rectangle(
                [int(50 * scale), yy - int(8 * scale), int(300 * scale), yy + int(30 * scale)],
                outline=ACCENT,
                width=max(1, int(2 * scale)),
            )
            draw.text((int(64 * scale), yy), line, font=font, fill=INK)
            if i < len(shown) - 1:
                ax = int(175 * scale)
                draw.line(
                    [(ax, yy + int(30 * scale)), (ax, yy + int(50 * scale))],
                    fill=ACCENT,
                    width=max(1, int(3 * scale)),
                )
                draw.polygon(
                    [
                        (ax - int(6 * scale), yy + int(46 * scale)),
                        (ax + int(6 * scale), yy + int(46 * scale)),
                        (ax, yy + int(56 * scale)),
                    ],
                    fill=ACCENT,
                )

    elif layout == "colored":
        palette = [(216, 232, 214), (244, 226, 206), (214, 224, 244), (240, 214, 220)]
        font = load_font(SANS, int(21 * scale))
        for i, line in enumerate(shown):
            yy = y + int(16 * scale) + i * int(60 * scale)
            draw.rectangle(
                [int(46 * scale), yy, width - int(46 * scale), yy + int(46 * scale)],
                fill=palette[i % len(palette)],
            )
            draw.text((int(62 * scale), yy + int(10 * scale)), line, font=font, fill=INK)

    else:  # plain
        font = load_font(SERIF, int(23 * scale))
        for i, line in enumerate(shown):
            draw.text(
                (int(50 * scale), y + int(20 * scale) + i * int(42 * scale)),
                line,
                font=font,
                fill=INK,
            )

    return img


# --------------------------------------------------------------------------
# 講者影格
# --------------------------------------------------------------------------


def render_speaker(size: tuple[int, int], seed: int = 0) -> Image.Image:
    """全螢幕講者（SDD §1.3：硬切，無 PiP）。

    畫得盡量接近一張人像照片：柔和漸層背景、正確比例的五官、光影、膠片
    雜訊。合成人臉是否騙得過真實的人臉偵測器，是 Phase 1 必須實測的事，
    見 docs/known-risks.md。
    """
    width, height = size
    rng = np.random.default_rng(seed)

    # 漸層背景（攝影棚打光感）
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width * 0.5, height * 0.35
    radial = np.sqrt(((xx - cx) / width) ** 2 + ((yy - cy) / height) ** 2)
    base = np.clip(1.0 - radial * 1.1, 0.15, 1.0)
    bg = np.dstack([base * 78, base * 88, base * 104]).astype(np.uint8)
    img = Image.fromarray(bg).filter(ImageFilter.GaussianBlur(6))
    draw = ImageDraw.Draw(img)

    s = height / 360.0
    face_h = height * 0.46
    face_w = face_h * 0.72
    fx, fy = width * 0.5, height * 0.40

    skin = (222, 184, 152)
    skin_dark = (196, 158, 128)
    hair = (46, 38, 36)

    # 肩膀與脖子
    draw.polygon(
        [
            (width * 0.5 - face_w * 1.5, height),
            (width * 0.5 - face_w * 0.62, fy + face_h * 0.44),
            (width * 0.5 + face_w * 0.62, fy + face_h * 0.44),
            (width * 0.5 + face_w * 1.5, height),
        ],
        fill=(58, 62, 78),
    )
    draw.rectangle(
        [fx - face_w * 0.20, fy + face_h * 0.28, fx + face_w * 0.20, fy + face_h * 0.60],
        fill=skin_dark,
    )

    # 臉
    draw.ellipse([fx - face_w / 2, fy - face_h / 2, fx + face_w / 2, fy + face_h / 2], fill=skin)
    # 頭髮
    draw.chord(
        [fx - face_w / 2 - 2 * s, fy - face_h / 2 - 6 * s, fx + face_w / 2 + 2 * s, fy + face_h * 0.10],
        180,
        360,
        fill=hair,
    )

    eye_dy = fy - face_h * 0.06
    eye_dx = face_w * 0.21
    eye_w, eye_h = face_w * 0.15, face_h * 0.055
    for sign in (-1, 1):
        ex = fx + sign * eye_dx
        draw.ellipse([ex - eye_w, eye_dy - eye_h, ex + eye_w, eye_dy + eye_h], fill=(248, 248, 246))
        draw.ellipse(
            [ex - eye_h * 0.85, eye_dy - eye_h * 0.85, ex + eye_h * 0.85, eye_dy + eye_h * 0.85],
            fill=(72, 54, 42),
        )
        draw.ellipse(
            [ex - eye_h * 0.38, eye_dy - eye_h * 0.38, ex + eye_h * 0.38, eye_dy + eye_h * 0.38],
            fill=(18, 16, 16),
        )
        # 眉
        draw.line(
            [
                (ex - eye_w * 1.05, eye_dy - eye_h * 2.6),
                (ex + eye_w * 1.05, eye_dy - eye_h * 3.1),
            ],
            fill=hair,
            width=max(2, int(3 * s)),
        )

    # 鼻（以陰影表現，避免畫成一條線）
    draw.polygon(
        [
            (fx, fy + face_h * 0.02),
            (fx - face_w * 0.075, fy + face_h * 0.14),
            (fx + face_w * 0.055, fy + face_h * 0.14),
        ],
        fill=skin_dark,
    )
    # 嘴
    draw.chord(
        [fx - face_w * 0.17, fy + face_h * 0.17, fx + face_w * 0.17, fy + face_h * 0.30],
        10,
        170,
        fill=(158, 92, 88),
    )

    img = img.filter(ImageFilter.GaussianBlur(0.7))

    # 膠片雜訊：真實影格不會是純平色塊，這對 frame-diff 類方法是必要的干擾
    arr = np.asarray(img).astype(np.int16)
    arr += rng.normal(0, 3.2, arr.shape).astype(np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def save(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path
