"""R8 疊加偵測實驗的共用工具。

**這是實驗程式碼，不是 pipeline 的一部分。** 不得被 src/weft/ 匯入。

前提：方向 1（背景距離 + Otsu 二分類）已決定採用，此處直接當作既有事實
使用，不再重測。本實驗要回答的是方向 2——疊加模式能不能偵測。

幀與時間的對應：`02_frames/f_%05d.png` 從 1 開始編號，第 i 個（0-based）
對應 t = i + 0.5 秒（1 fps，取每格中心）。這與 `weft.stages.frames`
的 `t=(i + 0.5) / fps` 一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work" / "zIglvjoU9vo"
FRAMES_DIR = WORK / "02_frames"
OUT = Path(__file__).resolve().parent

#: 分類用的縮圖尺寸。夠小才能把 2519 張一次載進記憶體算中位數，
#: 又夠大到能分出「攝影棚」與「全螢幕投影片」。
TINY = (64, 36)

#: 殘差分析用的尺寸。要看得到疊加的文字，所以比 TINY 大得多；
#: 與 pipeline 的 downscale_short_side=180 對齊，讓 H2 的結論可直接套用。
WORK_SHORT_SIDE = 180


def frame_paths() -> list[Path]:
    return sorted(FRAMES_DIR.glob("f_*.png"))


def index_to_time(i: int) -> float:
    return i + 0.5


def time_to_index(t: float) -> int:
    return int(t - 0.5) if t >= 0.5 else 0


def load_tiny(paths: list[Path]) -> np.ndarray:
    """把全片載成 (N, 36, 64) 的灰階陣列。"""
    out = np.empty((len(paths), TINY[1], TINY[0]), dtype=np.float32)
    for i, p in enumerate(paths):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        out[i] = cv2.resize(g, TINY, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return out


def otsu_cut(values: np.ndarray) -> float:
    """對一維數值找 Otsu 門檻。與 pipeline 的 ink 遮罩同手法（decisions.md D7）。"""
    span = float(values.max())
    if span <= 0:
        return 0.0
    u8 = np.clip(values / span * 255.0, 0, 255).astype(np.uint8)
    level, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(level) / 255.0 * span


@dataclass(frozen=True)
class Split:
    """方向 1 的分類結果。"""

    distance: np.ndarray  # 每幀與攝影棚常態的距離
    cut: float
    is_fullscreen: np.ndarray  # True = 全螢幕投影片／片頭片尾

    @property
    def studio_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.is_fullscreen)


def split_studio_vs_fullscreen(tiny: np.ndarray) -> Split:
    """方向 1：與全片中位幀的距離 + Otsu。已決定採用，此處只是重現。"""
    reference = np.median(tiny, axis=0)
    distance = np.mean(np.abs(tiny - reference), axis=(1, 2))
    cut = otsu_cut(distance)
    return Split(distance=distance, cut=cut, is_fullscreen=distance > cut)


def load_work_gray(path: Path) -> np.ndarray:
    """載入單幀並縮到 WORK_SHORT_SIDE，回傳 [0,1] 的灰階。

    **不做高斯模糊**——pipeline 的模糊是為了壓制雷射筆與壓縮雜訊，
    但殘差分析要看的正是「相對於基準的差異」，先模糊會把疊加的細字磨掉。
    模糊與否對結論的影響會在報告中一併量測。
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    scale = WORK_SHORT_SIDE / min(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def save_json(name: str, payload) -> Path:
    path = OUT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))
