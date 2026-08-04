"""逐字稿與投影片的對齊。SDD §4.6。

「這一步**不呼叫 LLM**，避免與 S4 形成循環依賴。」

流程：投影片切換時間戳粗切 → embedding 邊界吸附 → segments。
吸附範圍**硬限制在 ±20 秒內**（§4.6 關鍵約束）——放大等同繞過設計約束，
所以上限在 `snap_boundary` 內部再夾一次，不只依賴設定值。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Window:
    """一段時間區間及其對應的投影片（speaker_only 時為 None）。"""

    t_start: float
    t_end: float
    slide_id: str | None


def coarse_windows(candidates, duration: float, min_segment_sec: float) -> list[Window]:
    """以投影片切換時間戳粗切。SDD §4.6 步驟 1。

    投影片之間的空隙即為純講者時段，自成一段（§4.6 步驟 4）。
    產生的區間**必須無縫覆蓋 [0, duration]**，否則 §5.3 不變量 2 會失敗。
    """
    windows: list[Window] = []
    cursor = 0.0
    for c in sorted(candidates, key=lambda x: x.t_start):
        if c.t_start - cursor > min_segment_sec:
            windows.append(Window(cursor, c.t_start, None))
        elif c.t_start > cursor and windows:
            # 空隙太短，併進前一段而不是留下破洞
            windows[-1] = Window(windows[-1].t_start, c.t_start, windows[-1].slide_id)
        windows.append(Window(max(cursor, c.t_start), c.t_end, f"slide_{c.index + 1:03d}"))
        cursor = c.t_end

    if duration - cursor > min_segment_sec or not windows:
        windows.append(Window(cursor, duration, None))
    elif duration > cursor:
        windows[-1] = Window(windows[-1].t_start, duration, windows[-1].slide_id)
    return windows


def snap_boundary(
    boundary: float,
    cues,
    prev_text: str,
    next_text: str,
    encode,
    window_sec: float,
) -> tuple[float, float]:
    """把邊界吸附到語意轉折點。SDD §4.6 步驟 2–3。回傳 `(新邊界, 位移)`。

    作法：取邊界前後 `window_sec` 內的句子，逐句計算它與「前一張投影片文字」
    和「後一張投影片文字」的相似度差；差值由正轉負之處即為語意轉折點。

    吸附範圍**硬限制在 ±20 秒**（§4.6 關鍵約束）。這裡對傳入的 window_sec
    再夾一次上限——設定檔調大它等同繞過設計約束，不該只靠自律。
    """
    limit = min(window_sec, HARD_SNAP_LIMIT_SEC)
    nearby = [c for c in cues if abs(c.t_start - boundary) <= limit]
    if len(nearby) < 2 or not prev_text.strip() or not next_text.strip():
        return boundary, 0.0

    texts = [c.text_corrected or c.text_raw for c in nearby]
    vectors = encode(texts + [prev_text, next_text])
    cue_vecs, prev_vec, next_vec = vectors[:-2], vectors[-2], vectors[-1]

    # 兩張投影片若語意上無法區分，「這句比較像哪一張」就沒有意義，吸附會
    # 退化成雜訊。實測 A1 的 slide_002／slide_003（兩張都是經文逐月列表）
    # 相似度 0.873，吸附把邊界從 90s 推到 70s——比不吸附還差 20 秒。
    # 見 docs/decisions.md D11。
    if float(prev_vec @ next_vec) >= MAX_SNAPPABLE_SLIDE_SIMILARITY:
        log.debug("相鄰投影片過於相似，略過 %0.1fs 的邊界吸附", boundary)
        return boundary, 0.0

    # 正值 = 這句比較像前一張投影片；負值 = 比較像後一張
    lean = cue_vecs @ prev_vec - cue_vecs @ next_vec

    # 找由正轉負的位置。找不到轉折 → **不吸附**。
    # 舊版在此退回序列端點，等於「沒有證據時採用最大位移」——正好相反。
    crossing = None
    for i in range(1, len(lean)):
        if lean[i - 1] > 0 >= lean[i]:
            crossing = i
            break
    if crossing is None:
        return boundary, 0.0

    snapped = nearby[crossing].t_start
    shift = snapped - boundary
    if abs(shift) > limit:  # 理論上不會發生，但這是硬約束，寧可夾住
        return boundary, 0.0
    return snapped, shift


#: SDD §4.6：「吸附範圍硬限制在 ±20 秒內。」
#: 寫成模組常數而非只讀設定，是為了讓「調大設定就能繞過」這件事不成立。
HARD_SNAP_LIMIT_SEC = 20.0

#: 相鄰投影片的 embedding 相似度超過此值就不吸附——語意上分不開的兩張圖，
#: 「這句比較像哪一張」是雜訊。實測值見 docs/decisions.md D11。
MAX_SNAPPABLE_SLIDE_SIMILARITY = 0.85


def assign_cues(windows: list[Window], cues) -> list[list[int]]:
    """把每句逐字稿指派給**恰好一個** segment。SDD §5.3 不變量 3。

    判準是句子的**中點**落在哪個區間——用起點會讓跨越邊界的長句全部歸給
    前一段，用終點則相反；中點對兩邊都公平。落在所有區間之外的（例如
    時間戳超出片長）歸給時間最近的一段，絕不丟棄。
    """
    buckets: list[list[int]] = [[] for _ in windows]
    for cue in cues:
        midpoint = (cue.t_start + cue.t_end) / 2.0
        placed = False
        for i, w in enumerate(windows):
            if w.t_start <= midpoint < w.t_end:
                buckets[i].append(cue.index)
                placed = True
                break
        if not placed:
            nearest = min(
                range(len(windows)),
                key=lambda i: min(abs(midpoint - windows[i].t_start), abs(midpoint - windows[i].t_end)),
            )
            buckets[nearest].append(cue.index)
    return buckets


class Encoder:
    """BGE-M3 的薄封裝。SDD §2.3：embedding 高頻，**絕不打 API**。"""

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
