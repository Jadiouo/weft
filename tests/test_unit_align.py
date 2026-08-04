"""對齊。SDD §4.6。

這裡測的是**結構性質**——粗切必須無縫覆蓋全片、每句逐字稿恰好屬於一個
segment、吸附不得超出 ±20 秒。這些一旦錯了，§5.3 的不變量 1/2/3 會全部
連鎖失敗，而且很難從最終產物反推是哪裡出的問題。

**v0.3 移除了語意邊界吸附**（見 known-risks R10），所以這裡只剩粗切與
指派的測試。
"""

from __future__ import annotations

import pytest

from weft.ir import SlideCandidate, TranscriptCue
from weft.stages.align import Window, assign_cues, coarse_windows


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
# §4.6：這一步不呼叫 LLM
# --------------------------------------------------------------------------


def test_align_does_not_call_an_llm():
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


def test_semantic_snap_is_gone():
    """v0.3 移除了語意邊界吸附。

    這條測試釘住「它是被**刻意**移除的，不是漏掉」——吸附需要投影片文字，
    而 v0.3 移除本地 OCR 後，投影片文字要到 S4 才有，§4.6 又禁止 S3 呼叫
    LLM。若日後有人想加回來，得先解決這個順序問題。見 known-risks R10。
    """
    import weft.stages.align as align

    assert not hasattr(align, "snap_boundary")
    assert not hasattr(align, "Encoder")
