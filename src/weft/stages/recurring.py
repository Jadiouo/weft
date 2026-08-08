"""跨集重現前濾——用「這一幀在別集也一模一樣出現過」剔除節目包裝。

SDD §4.7a 的判準是「畫面上有沒有**這一集特有的**講解內容」。
邏輯上它涵蓋全部負例：片頭字卡、頻道 logo、系列名稱卡、主講人學經歷、
片尾訂閱畫面——它們的共同點正是**每集都一樣**。

**但模型一次只看得到一張圖**，「別集長什麼樣」不在它的輸入裡。
R26 實測：分類 prompt 改到第五版，錯誤仍集中在片頭那幾張。

跨集比對不需要模型。實測（四集、80 張代表幀）：

| | 跨集最小灰階 MAE |
|---|---|
| 是投影片（n=26） | 最小 **7.57** |
| 不是（n=54） | 最小 0.44、中位 3.61 |

門檻取 6.0 時抓到 29/54 個包裝幀，**誤殺投影片 0/26**。
接在分類前面當前濾，四集合計 0.900 → **0.925**。

**代價：這一步讓 S4a 依賴「同系列其他集也處理過」**，
而 S-1／S1c 是刻意設計成逐集獨立的。因此它是明確的設定開關，
而且參考集不足時**明說並跳過**，不靜默降級（§5.5 #6 的精神）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: 與 §4.3 同一個短邊。**灰階、不模糊**——D27 量過模糊會誤併橫幅標題卡。
SHORT_SIDE = 180
#: 至少要有這麼多支參考影片才跑。一支的話「跨集」只是「跨這一支」，
#: 撞到同系列同一段素材的機率高，分離度也沒量過。
MIN_REFERENCE_VIDEOS = 2


def _load(path: Path) -> np.ndarray | None:
    from PIL import Image

    try:
        im = Image.open(path).convert("L")
    except Exception:  # noqa: BLE001 —— 壞掉的圖當作沒有
        return None
    w, h = im.size
    s = SHORT_SIDE / min(w, h)
    return np.asarray(im.resize((int(w * s), int(h * s))), dtype=np.float32)


def reference_frames(work_dir: Path, exclude_video_id: str) -> list[np.ndarray]:
    """其他影片的代表幀。沒有 `04_dedup.json` 的影片直接跳過。

    用**代表幀**而不是全部候選幀：同一段包裝在一支影片內會出現很多次，
    全讀進來只是拖慢，分離度不會變。
    """
    import json

    out: list[np.ndarray] = []
    for video_dir in sorted(p for p in work_dir.iterdir() if p.is_dir()):
        if video_dir.name == exclude_video_id:
            continue
        dedup = video_dir / "04_dedup.json"
        slides_dir = video_dir / "03_slides"
        if not dedup.exists() or not slides_dir.is_dir():
            continue
        try:
            entries = json.loads(dedup.read_text(encoding="utf-8"))["slides"]
        except Exception:  # noqa: BLE001
            continue
        reps = {sid for sid, v in entries.items() if not v.get("duplicate_of")}
        for sid in sorted(reps):
            arr = _load(slides_dir / f"{sid}.png")
            if arr is not None:
                out.append(arr)
    return out


def recurring_slide_ids(work, slides, threshold: float, references) -> dict[str, float]:
    """`slide_id → 跨集最小 MAE`，只含低於門檻的（＝判定為節目包裝）。"""
    hits: dict[str, float] = {}
    for slide in slides:
        arr = _load(work.dir / slide.image_path)
        if arr is None:
            continue
        distances = [float(np.abs(arr - ref).mean()) for ref in references
                     if ref.shape == arr.shape]
        if not distances:
            continue
        best = min(distances)
        if best < threshold:
            hits[slide.slide_id] = best
    return hits


def count_reference_videos(work_dir: Path, exclude_video_id: str) -> int:
    return sum(
        1 for p in work_dir.iterdir()
        if p.is_dir() and p.name != exclude_video_id and (p / "04_dedup.json").exists()
    )
