"""對齊。SDD §4.6。

這裡測的是**結構性質**——粗切必須無縫覆蓋全片、每句逐字稿恰好屬於一個
segment、吸附不得超出 ±20 秒。這些一旦錯了，§5.3 的不變量 1/2/3 會全部
連鎖失敗，而且很難從最終產物反推是哪裡出的問題。

語意吸附的**品質**（誤差中位數 ≤ 5 秒）需要真實影片黃金集，見
`test_e2e_pipeline.py` 與 `docs/known-risks.md` R2。
"""

from __future__ import annotations

import numpy as np
import pytest

from weft.ir import SlideCandidate, TranscriptCue
from weft.stages.align import (
    HARD_SNAP_LIMIT_SEC,
    Window,
    assign_cues,
    coarse_windows,
    snap_boundary,
)


def candidate(index: int, t_start: float, t_end: float) -> SlideCandidate:
    return SlideCandidate(index=index, t_start=t_start, t_end=t_end, keyframe_t=t_end - 0.5)


def cue(index: int, t_start: float, t_end: float, text: str = "測試") -> TranscriptCue:
    return TranscriptCue(index=index, t_start=t_start, t_end=t_end, text_raw=text)


# --------------------------------------------------------------------------
# 粗切（§4.6 步驟 1、4）
# --------------------------------------------------------------------------


def test_windows_cover_the_whole_video_without_gaps():
    """§5.3 不變量 2：segments 聯集等於影片全長。粗切階段就必須成立。"""
    windows = coarse_windows([candidate(0, 30, 90), candidate(1, 120, 200)], 250.0, 5.0)

    assert windows[0].t_start == 0.0
    assert windows[-1].t_end == 250.0
    for prev, cur in zip(windows, windows[1:]):
        assert cur.t_start == pytest.approx(prev.t_end), "粗切留下了空隙"


def test_windows_do_not_overlap():
    """§5.3 不變量 1。"""
    windows = coarse_windows([candidate(0, 30, 90), candidate(1, 120, 200)], 250.0, 5.0)
    for prev, cur in zip(windows, windows[1:]):
        assert cur.t_start >= prev.t_end


def test_gaps_between_slides_become_speaker_only():
    """§4.6 步驟 4：純講者時段自成 segment。"""
    windows = coarse_windows([candidate(0, 30, 90), candidate(1, 120, 200)], 250.0, 5.0)
    speaker = [w for w in windows if w.slide_id is None]
    assert len(speaker) == 3  # 開頭 [0,30)、中間 [90,120)、結尾 [200,250)


def test_short_gaps_are_absorbed_not_left_as_holes():
    """短於 min_segment_sec 的空隙併進前一段——留成獨立 segment 會產生
    一堆 1 秒的碎片，留成空洞則違反不變量 2。"""
    windows = coarse_windows([candidate(0, 0, 90), candidate(1, 92, 200)], 200.0, 5.0)
    assert len(windows) == 2
    assert windows[0].t_end == 92.0


def test_no_slides_yields_single_window():
    """對抗樣本 A3：全片無投影片 → 整片一段（transcript_only 模式）。"""
    windows = coarse_windows([], 120.0, 5.0)
    assert windows == [Window(0.0, 120.0, None)]


def test_slide_covering_whole_video_yields_one_window():
    """對抗樣本 A4：整片就一張投影片。"""
    windows = coarse_windows([candidate(0, 0, 90)], 90.0, 5.0)
    assert len(windows) == 1
    assert windows[0].slide_id == "slide_001"


# --------------------------------------------------------------------------
# 逐字稿指派（§5.3 不變量 3）
# --------------------------------------------------------------------------


def test_every_cue_is_assigned_exactly_once():
    windows = coarse_windows([candidate(0, 30, 90)], 120.0, 5.0)
    cues = [cue(i, i * 10.0, i * 10.0 + 9.0) for i in range(12)]

    buckets = assign_cues(windows, cues)
    flat = [i for b in buckets for i in b]

    assert sorted(flat) == [c.index for c in cues]
    assert len(flat) == len(set(flat)), "有句子被指派到多個 segment"


def test_cue_is_assigned_by_its_midpoint():
    """跨越邊界的長句：用起點會全歸前段，用終點會全歸後段；中點對兩邊公平。"""
    windows = [Window(0.0, 50.0, "slide_001"), Window(50.0, 100.0, "slide_002")]
    # 這句 [40, 70) 的中點 55 落在後段
    buckets = assign_cues(windows, [cue(0, 40.0, 70.0)])
    assert buckets == [[], [0]]


def test_out_of_range_cue_goes_to_nearest_window_not_dropped():
    """時間戳超出片長的句子（字幕檔常見）不得被丟棄——
    丟一句就違反不變量 3。"""
    windows = [Window(0.0, 100.0, None)]
    buckets = assign_cues(windows, [cue(0, 105.0, 110.0)])
    assert buckets == [[0]]


def test_empty_transcript_yields_empty_buckets():
    windows = coarse_windows([candidate(0, 0, 90)], 90.0, 5.0)
    assert assign_cues(windows, []) == [[]]


# --------------------------------------------------------------------------
# 語意吸附（§4.6 步驟 2–3、關鍵約束）
# --------------------------------------------------------------------------


def fake_encoder(mapping: dict[str, np.ndarray]):
    """把文字映到指定向量。這是**單元測試**（檔名含 `_unit_`），
    §5.5 #10 允許；e2e 用真的 BGE-M3。"""

    def encode(texts: list[str]) -> np.ndarray:
        return np.stack([mapping[t] for t in texts])

    return encode


PREV = np.array([1.0, 0.0])
NEXT = np.array([0.0, 1.0])


def test_boundary_snaps_to_the_semantic_turning_point():
    """講者在投影片切換**之前**就講完上一頁——邊界該往前吸。"""
    cues = [
        cue(0, 90.0, 95.0, "前頁內容甲"),
        cue(1, 95.0, 100.0, "前頁內容乙"),
        cue(2, 100.0, 105.0, "後頁內容甲"),  # 語意在此轉折
        cue(3, 105.0, 110.0, "後頁內容乙"),
    ]
    encode = fake_encoder({
        "前頁內容甲": PREV, "前頁內容乙": PREV,
        "後頁內容甲": NEXT, "後頁內容乙": NEXT,
        "前頁文字": PREV, "後頁文字": NEXT,
    })

    snapped, shift = snap_boundary(108.0, cues, "前頁文字", "後頁文字", encode, 20.0)
    assert snapped == 100.0
    assert shift == pytest.approx(-8.0)


def test_snap_is_hard_limited_to_twenty_seconds():
    """§4.6 關鍵約束：吸附範圍**硬限制在 ±20 秒內**。

    這條測試刻意傳入一個過大的 window——設定檔調大它不該就能繞過約束。
    """
    cues = [cue(i, 100.0 + i * 20.0, 115.0 + i * 20.0, f"句{i}") for i in range(8)]
    encode = fake_encoder({**{f"句{i}": (PREV if i < 6 else NEXT) for i in range(8)},
                           "前": PREV, "後": NEXT})

    snapped, shift = snap_boundary(140.0, cues, "前", "後", encode, window_sec=999.0)
    assert abs(shift) <= HARD_SNAP_LIMIT_SEC


def test_hard_limit_matches_sdd():
    assert HARD_SNAP_LIMIT_SEC == 20.0


def test_snap_is_skipped_without_slide_text():
    """沒有 OCR 文字就無從比較語意，維持粗切。這是降級，不是失敗。"""
    cues = [cue(i, 90.0 + i * 5.0, 95.0 + i * 5.0, f"句{i}") for i in range(4)]
    snapped, shift = snap_boundary(100.0, cues, "", "後頁文字", lambda t: np.zeros((len(t), 2)), 20.0)
    assert (snapped, shift) == (100.0, 0.0)


def test_snap_is_skipped_with_too_few_nearby_cues():
    snapped, shift = snap_boundary(100.0, [cue(0, 99.0, 101.0)], "前", "後",
                                   lambda t: np.zeros((len(t), 2)), 20.0)
    assert (snapped, shift) == (100.0, 0.0)


def test_snap_does_not_call_an_llm():
    """§4.6：「這一步**不呼叫 LLM**，避免與 S4 形成循環依賴。」

    機械式護欄：align 模組不得引入任何雲端模型客戶端。
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src/weft/stages/align.py").read_text(
        encoding="utf-8"
    )
    forbidden = {"google", "genai", "openai", "anthropic", "requests", "httpx"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {(getattr(node, "module", "") or "").split(".")[0]}
            names |= {a.name.split(".")[0] for a in node.names}
            assert not (names & forbidden), f"align.py 引入了雲端客戶端：{names & forbidden}"
