"""逐字稿與候選區段的對齊。SDD §4.6。

「這一步**不呼叫 LLM**，避免與 S4 形成循環依賴。」

**v0.3 移除了語意邊界吸附。** 吸附需要投影片文字來判斷「這句話比較像前一張
還是後一張」；v0.3 拿掉本地 OCR 後，投影片文字要到 S4 才有，而 §4.6 禁止
S3 呼叫 LLM。S3 現在只做粗切——區段邊界直接來自 S1b 的靜止區段偵測。

這是這次簡化的**已知代價**，記在 known-risks R10。實測上損失有限：真實素材
的相鄰投影片語意相似度達 0.873，吸附本來就會被 D11 的守門擋掉而不作用。
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


