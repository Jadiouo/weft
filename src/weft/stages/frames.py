"""抽幀與幀特徵。SDD §4.3 步驟 1–3。

這些是 S1b 的前置：把影片變成一串「已降噪」的特徵向量，靜止區段偵測
（detect.py）只在這串向量上工作，不再碰影像。分開的理由是偵測的參數
要能快速反覆實驗，而抽幀很慢。

**v0.3 移除了 speaker/slide 二分類。** 原本靠偵測滿版人臉，實測在真實素材
上失效（人臉面積佔比 0.005–0.026 vs 門檻 0.04；而投影片中的人像會被誤測成
0.024，與講者畫面完全重疊）。現在 CV 只負責「找出畫面靜止的區段」，
「這張圖是不是投影片」交給 S4 的 VLM 判斷——它本來就要看這張圖。
見 docs/decisions.md D16。
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    """一個抽樣點。`feature` 是降解析度＋模糊後的灰階圖。"""

    index: int
    t: float
    feature: np.ndarray
    #: 前景遮罩（非背景像素）。靜止區段偵測與逐條動畫合併都用它。
    ink: np.ndarray
    path: Path | None = None


def extract_frames(video: Path, out_dir: Path, fps: float) -> list[Path]:
    """ffmpeg 每 1/fps 秒抽一幀。SDD §4.3 步驟 1。

    落地成 PNG 而非留在記憶體：一支 42 分鐘影片在 1fps 下是 2520 張，
    以 180px 縮圖計不到 100MB，而落地讓抽幀本身可以獨立重跑。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("f_*.png"))
    if existing:
        return existing

    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         # 取每格中心，與 SynthTruth.frame_classes 的抽樣點對齊
         "-vf", f"fps={fps}:round=down",
         "-fps_mode", "passthrough",
         str(out_dir / "f_%05d.png")],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"抽幀失敗：{proc.stderr[-1500:]}")
    return sorted(out_dir.glob("f_*.png"))


def _downscale_and_blur(img: np.ndarray, short_side: int, sigma: float) -> np.ndarray:
    """SDD §4.3 步驟 3：降解析度 + 高斯模糊。

    目的是壓制雷射筆紅點與壓縮雜訊（對抗樣本 A4）。降解析度先做——
    先模糊再縮小等於做了兩次低通，會把換頁該有的訊號也磨掉。
    """
    h, w = img.shape[:2]
    scale = short_side / min(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    return gray.astype(np.float32) / 255.0


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """前景（文字／圖形）遮罩。

    投影片背景是大面積單一色，所以取「偏離背景亮度」的像素即可，不需要
    真的做文字偵測。用於換頁偵測與逐條動畫的視覺包含判斷（見 detect.py）。

    門檻用 **Otsu**（找前景／背景兩群之間的谷底），不用「某分位數的固定
    比例」。差別是實測出來的：後者在色塊版型上，淺色區塊的亮度恰好落在
    門檻附近，投影機／會場燈光的亮度浮動就會讓整塊翻進翻出，ink 量在
    6600↔9040 之間震盪，把非邊界的 Jaccard 推到 0.27。改用 Otsu 後
    降到 0.025，分離倍數從 2.4x 拉到 27.5x。見 docs/decisions.md D7。
    """
    deviation = np.abs(gray - float(np.median(gray)))
    peak = float(deviation.max())
    if peak < 1e-3:
        return np.zeros_like(gray, dtype=bool)  # 全白／全黑畫面
    quantised = np.clip(deviation / peak * 255.0, 0, 255).astype(np.uint8)
    level, _ = cv2.threshold(quantised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return deviation > (level / 255.0 * peak)


def load_frames(paths: list[Path], fps: float, short_side: int, sigma: float) -> list[Frame]:
    frames: list[Frame] = []
    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"讀不到抽出的幀：{path}")
        gray = _downscale_and_blur(img, short_side, sigma)
        frames.append(
            Frame(
                index=i,
                t=(i + 0.5) / fps,  # 每格中心
                feature=gray,
                ink=_ink_mask(gray),
                path=path,
            )
        )
    return frames


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    """兩幀之間的距離，落在 [0, 1]。

    用平均絕對差而非 SSIM 或直方圖：換頁是**整版替換**，平均絕對差對此
    最敏感；而直方圖對「同色系不同版面」幾乎無感（例如 A7 回放的兩張
    投影片底色相同）。
    """
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(a - b)))


def consecutive_distances(frames: list[Frame]) -> np.ndarray:
    """d[i] = 第 i 幀與第 i-1 幀的距離；d[0] = 0。"""
    d = np.zeros(len(frames), dtype=np.float64)
    for i in range(1, len(frames)):
        d[i] = frame_distance(frames[i - 1].feature, frames[i].feature)
    return d


def ink_containment(earlier: np.ndarray, later: np.ndarray) -> float:
    """earlier 的前景像素有多少比例仍存在於 later 中。

    逐條動畫的判準（SDD §4.3 步驟 5）：新的一段是舊的一段**加上**新內容，
    舊內容位置不變 → containment ≈ 1。換到不同頁 → 版面全變 → containment 低。
    """
    if not earlier.any():
        return 1.0  # 空白頁被任何內容「包含」
    if earlier.shape != later.shape:
        later = cv2.resize(
            later.astype(np.uint8), (earlier.shape[1], earlier.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return float(np.logical_and(earlier, later).sum() / earlier.sum())
