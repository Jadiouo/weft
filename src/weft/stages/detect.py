"""換頁偵測與逐條動畫合併。SDD §4.3 步驟 4–6。

觀測量是 **ink Jaccard 距離**——前景（文字／圖形）遮罩的變化比例——而不是
原始像素差。理由是實測的（見 docs/decisions.md D7）：原始像素距離跨影片
相差 100 倍（A2 的真實邊界 0.007，A6 的 0.72），必須逐片正規化；ink Jaccard
本身就落在 [0,1] 且語意固定（「ink 圖樣有多少比例變了」），真實邊界 ≥0.63、
非邊界 ≤0.22，不需要正規化就分得開。

邊界有兩個來源，缺一不可：
  (a) slide↔speaker 的切換    ← 由分類決定，A6 的四個邊界全部來自這裡
  (b) 連續 slide 段落內的換頁  ← 由本模組的 HMM 決定
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .frames import Frame, ink_containment

log = logging.getLogger(__name__)

SAME, CHANGE = 0, 1

#: 估計「同一頁」離散程度時採用的分位數。高於此分位的值視為可能的換頁，
#: 不納入基線估計。0.75 等於假設換頁幀不超過四分之一——在 1fps 下相當於
#: 每 4 秒換一頁，遠比任何真實講課密集。
BULK_QUANTILE = 0.75


@dataclass(frozen=True)
class Run:
    """一段連續的同類幀（全 slide 或全 speaker）。"""

    is_speaker: bool
    start: int  # frame index，含
    end: int  # frame index，不含

    def frames(self, frames: list[Frame]) -> list[Frame]:
        return frames[self.start : self.end]


def split_runs(frames: list[Frame]) -> list[Run]:
    """把幀序列切成 speaker／slide 交錯的段落。"""
    runs: list[Run] = []
    if not frames:
        return runs
    start, kind = 0, frames[0].is_speaker
    for i, f in enumerate(frames[1:], start=1):
        if f.is_speaker != kind:
            runs.append(Run(kind, start, i))
            start, kind = i, f.is_speaker
    runs.append(Run(kind, start, len(frames)))
    return runs


def ink_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """兩張 ink 遮罩的 Jaccard 距離。0 = 完全相同，1 = 毫無交集。"""
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0  # 兩張都是空白頁
    return 1.0 - float(np.logical_and(a, b).sum() / union)


def viterbi_changes(
    distances: np.ndarray,
    self_transition: float,
    min_ink_change: float,
) -> list[int]:
    """在距離序列上跑 2 狀態 HMM，回傳被判為換頁的 index。

    SDD §4.3 步驟 4：「以『投影片會停留一段時間』為先驗建模，避免手調門檻。」

    **狀態**：SAME（沿用同一頁）／CHANGE（此刻換頁）。
    `CHANGE → CHANGE` 的機率為 0——換頁是瞬間事件，不可能連續兩幀都在換。
    這一條是模型的關鍵：它讓 Viterbi 無法用「每幀都換」來解釋高頻抖動
    （例如 A5 投影片內嵌播放的影片），而必須整段判為 SAME。

    **先驗**：`self_transition = 0.97` 對應幾何分布的平均停留 ~33 幀，
    在 1fps 下即 ~33 秒，與 SDD §5.1 的「每頁停留 30–120 秒」相符。

    **發射**：SAME 的尺度由**資料自身**的離散程度估計（低於 75 分位者的
    平均），因此會隨影片自適應——投影片內嵌影片、壓縮雜訊多的片源，基線
    自然抬高。CHANGE 用 [0,1] 上的均勻分布，即「無資訊的離群假設」。

    唯一的常數 `min_ink_change` 是基線的下限，語意為「換頁至少會改變這個
    比例的 ink 圖樣」。它作用在**尺度無關**的 Jaccard 上，不是像素門檻。
    沒有它，一支完全沒換頁的影片（A4）會把量化雜訊當成離群值。
    """
    n = len(distances)
    if n == 0:
        return []

    bulk_cut = float(np.quantile(distances, BULK_QUANTILE))
    bulk = distances[distances <= bulk_cut]
    scale = max(float(bulk.mean()) if bulk.size else 0.0, min_ink_change)
    rate = 1.0 / scale

    # log 發射機率。CHANGE 為 Uniform[0,1] → log density = 0
    emit_same = np.log(rate) - rate * distances
    emit_change = np.zeros(n)

    log_stay = np.log(self_transition)
    log_leave = np.log1p(-self_transition)

    NEG_INF = -1e18
    delta = np.full((n, 2), NEG_INF)
    psi = np.zeros((n, 2), dtype=np.int8)
    delta[0, SAME] = emit_same[0]
    delta[0, CHANGE] = emit_change[0] + log_leave  # 第一幀就換頁不合理，但不禁止

    for t in range(1, n):
        # → SAME：可從 SAME 續留，或從 CHANGE 回到穩定
        cand = (delta[t - 1, SAME] + log_stay, delta[t - 1, CHANGE] + 0.0)
        psi[t, SAME] = int(cand[1] > cand[0])
        delta[t, SAME] = max(cand) + emit_same[t]
        # → CHANGE：只能從 SAME 來（CHANGE→CHANGE 機率為 0）
        psi[t, CHANGE] = SAME
        delta[t, CHANGE] = delta[t - 1, SAME] + log_leave + emit_change[t]

    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(delta[-1, CHANGE] > delta[-1, SAME])
    for t in range(n - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return [int(i) for i in np.flatnonzero(path == CHANGE)]


@dataclass
class Section:
    """一段被判定為「同一張投影片」的時間區間。"""

    start: int  # frame index，含
    end: int  # frame index，不含
    build_indices: list[int]  # 被合併掉的各 build 起點（frame index）

    @property
    def keyframe(self) -> int:
        """代表幀。逐條動畫時為**最後一幀**（內容最完整），SDD §4.3 步驟 5。"""
        return self.end - 1

    @property
    def is_progressive(self) -> bool:
        return len(self.build_indices) > 1


def merge_progressive(
    sections: list[Section],
    frames: list[Frame],
    containment_ratio: float,
) -> list[Section]:
    """把逐條動畫的各個 build 合併回一張投影片。SDD §4.3 步驟 5。

    判準是**視覺包含**：後一段的 ink 是否涵蓋前一段的 ink。

    SDD 原文寫的是「新段落的 OCR 文字包含舊段落的」，但 S2（OCR）在資料流上
    位於 S1b **之後**（§2.2），S1b 用不到 OCR 結果；且 §8 把 S1b 放在
    `pipe-cpu`、PaddleOCR 放在 `pipe-gpu`。視覺包含達成同樣的語意判斷，
    且不倒轉依賴、不需要 GPU。詳見 docs/decisions.md D8。

    實測分離（合成素材）：build 的 containment 0.86–0.98，真正換頁 ≤0.56。
    """
    if not sections:
        return []

    merged = [sections[0]]
    for section in sections[1:]:
        prev = merged[-1]
        # 拿兩段的代表幀比對：前一段最完整的狀態 vs 這一段最完整的狀態
        overlap = ink_containment(
            frames[prev.keyframe].ink, frames[section.keyframe].ink
        )
        if overlap >= containment_ratio:
            merged[-1] = Section(
                start=prev.start,
                end=section.end,
                build_indices=prev.build_indices + section.build_indices,
            )
        else:
            merged.append(section)
    return merged


def detect_sections(
    frames: list[Frame],
    run: Run,
    self_transition: float,
    min_ink_change: float,
    containment_ratio: float,
    merge_progressive_builds: bool,
) -> list[Section]:
    """在一段連續的 slide 幀中找出各張投影片。"""
    block = run.frames(frames)
    if not block:
        return []

    distances = np.zeros(len(block))
    for i in range(1, len(block)):
        distances[i] = ink_jaccard(block[i - 1].ink, block[i].ink)

    change_at = viterbi_changes(distances, self_transition, min_ink_change)

    starts = [0] + [i for i in change_at if i > 0]
    sections = [
        Section(
            start=run.start + s,
            end=run.start + (starts[k + 1] if k + 1 < len(starts) else len(block)),
            build_indices=[run.start + s],
        )
        for k, s in enumerate(starts)
    ]

    if merge_progressive_builds:
        sections = merge_progressive(sections, frames, containment_ratio)
    return sections


def drop_short_sections(
    sections: list[Section], frames: list[Frame], min_duration_sec: float, fps: float
) -> list[Section]:
    """濾掉過短的段落，並把時間併回前一段。

    過短的段落多半是轉場動畫或編碼瑕疵，不是真的投影片。**併入**而非丟棄，
    是為了維持 §5.3 不變量 2（segments 聯集等於影片全長）。
    """
    if not sections:
        return []
    min_frames = max(1, int(round(min_duration_sec * fps)))
    kept = [sections[0]]
    for section in sections[1:]:
        if section.end - section.start < min_frames:
            kept[-1] = Section(
                start=kept[-1].start,
                end=section.end,
                build_indices=kept[-1].build_indices + section.build_indices,
            )
        else:
            kept.append(section)
    return kept
