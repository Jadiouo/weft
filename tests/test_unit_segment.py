"""S3 分段——以逐字稿為主幹（票 08）。

測的是**外部行為**：哪些話被切在同一段、切點落在哪裡、
不變量還成不成立。不測內部的相似度矩陣長什麼樣。
"""

from __future__ import annotations

import numpy as np
import pytest

from weft.stages.segment import (
    char_ngram_vectors,
    depth_cut_indices,
    enforce_min_length,
    topic_boundaries,
    window_similarity,
)


class _Cue:
    def __init__(self, t_start: float, t_end: float, text: str):
        self.t_start, self.t_end, self.text_raw = t_start, t_end, text


def _cues(topics: list[tuple[str, int]], sec: float = 3.0) -> list[_Cue]:
    """把 `(句子, 重複次數)` 展成 cue 序列。同一個 topic 內用詞高度重疊。"""
    out, t = [], 0.0
    for text, n in topics:
        for _ in range(n):
            out.append(_Cue(t, t + sec, text))
            t += sec
    return out


#: 兩個用詞完全不重疊的話題。**這是分段最基本該做到的事**——
#: 連這個都切不出來的方法，在真實素材上更不可能。
_TWO_TOPICS = [
    ("胰腺是一個倉庫它要製造消化液送給小腸吸收各種東西", 8),
    ("肝臟每天晚上做盤點統計血球數量報表送給大腦決策", 8),
]


def test_boundary_lands_between_two_unrelated_topics():
    cues = _cues(_TWO_TOPICS)
    switch = cues[8].t_start
    found = topic_boundaries(cues, block_chars=24, window=2)
    assert found, "兩個完全不相干的話題之間一刀都沒切"
    assert min(abs(t - switch) for t in found) <= 12.0, (
        f"最近的切點離真正的話題轉換 {min(abs(t - switch) for t in found):.0f} 秒"
    )


def test_uniform_content_is_not_over_segmented():
    """整段講同一件事時不該被切碎。

    沒有這一條，上面那條可以靠「每個 block 都切」通過。
    """
    cues = _cues([("胰腺是一個倉庫它要製造消化液送給小腸吸收各種東西", 16)])
    assert len(topic_boundaries(cues, block_chars=24, window=2)) <= 1


def test_too_short_transcript_yields_no_boundaries():
    """內容太短時回傳空清單——那不是錯誤，是沒有東西可以據以分段。"""
    assert topic_boundaries(_cues([("很短", 2)]), block_chars=40, window=3) == []
    assert topic_boundaries([], block_chars=40, window=3) == []


def test_char_ngram_needs_no_tokenizer():
    """中文沒有詞邊界，所以表示層用字元共現。"""
    vecs = char_ngram_vectors(["胰腺倉庫", "胰腺倉庫", "肝臟盤點"])
    assert vecs.shape[0] == 3
    same = float(vecs[0] @ vecs[1])
    diff = float(vecs[0] @ vecs[2])
    assert same > diff, "同樣的文字應該比不同的文字更相似"


def test_window_similarity_dips_at_the_seam():
    """相似度序列在接縫處要低於兩側——TextTiling 靠的就是這個。"""
    vecs = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5)
    scores = window_similarity(vecs, window=2)
    assert scores.argmin() == 4, f"谷底應該在 index 4，實際在 {scores.argmin()}"


def test_depth_cut_needs_no_ground_truth_count():
    """斷點數量由深度分數自己決定，**不把正確答案的數量餵給它**。

    餵進去的話，任何方法都能「切 N 刀」，量到的就不是方法好不好。
    """
    scores = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9])
    cuts = depth_cut_indices(scores)
    assert cuts == [2, 6], cuts


def test_min_length_is_greedy_and_deterministic():
    """過短的區段要被丟掉，而且同樣的輸入永遠給同樣的輸出。"""
    got = enforce_min_length([10.0, 12.0, 50.0, 52.0], duration=100.0, min_sec=20.0)
    assert got == [50.0]
    assert enforce_min_length([10.0, 12.0, 50.0, 52.0], 100.0, 20.0) == got


def test_min_length_never_leaves_a_short_tail():
    """最後一段也不能過短——不變量 2 要求聯集等於全長，一個 1 秒的尾巴
    會是合法但無用的 chunk。"""
    assert enforce_min_length([95.0], duration=100.0, min_sec=20.0) == []


# --------------------------------------------------------------------------
# 投影片降級為註記（不參與決定切在哪裡）
# --------------------------------------------------------------------------


class _Cand:
    def __init__(self, index: int, t_start: float, t_end: float):
        self.index, self.t_start, self.t_end = index, t_start, t_end


def test_slide_is_annotation_not_driver():
    """切點來自話題邊界；投影片只回答「這一段螢幕上是哪一張」。"""
    from weft.stages.align import topic_windows

    # 一張投影片橫跨整段，而話題在中間換了
    windows = topic_windows([50.0], duration=100.0, candidates=[_Cand(0, 0.0, 100.0)])
    assert [(w.t_start, w.t_end) for w in windows] == [(0.0, 50.0), (50.0, 100.0)]
    assert [w.slide_id for w in windows] == ["slide_001", "slide_001"], (
        "同一張圖上講完兩件事時，兩段都該指向那張圖"
    )


def test_slide_with_most_overlap_wins():
    """一段話橫跨兩張圖時取重疊最久的，不是取第一張。"""
    from weft.stages.align import topic_windows

    windows = topic_windows([], duration=100.0,
                            candidates=[_Cand(0, 0.0, 20.0), _Cand(1, 20.0, 100.0)])
    assert windows[0].slide_id == "slide_002"


def test_no_overlap_means_no_slide_ref():
    """沒有畫面可指時就是 None——硬指一張等於給 §5.4 一個假的來源。"""
    from weft.stages.align import topic_windows

    windows = topic_windows([], duration=100.0, candidates=[])
    assert windows[0].slide_id is None


@pytest.mark.parametrize("boundaries", [[], [30.0], [30.0, 60.0]])
def test_windows_tile_the_whole_video(boundaries):
    """§5.3 不變量 1–2：區間不重疊、聯集等於全長。"""
    from weft.stages.align import topic_windows

    windows = topic_windows(boundaries, duration=90.0, candidates=[])
    assert windows[0].t_start == 0.0
    assert windows[-1].t_end == 90.0
    for a, b in zip(windows, windows[1:], strict=False):
        assert a.t_end == b.t_start


# --------------------------------------------------------------------------
# 冪等鍵必須涵蓋分段本身（票 08 踩到的坑）
# --------------------------------------------------------------------------


def _segment(segment_id: str, t_start: float, t_end: float, text: str,
             slide_ref: str | None = None):
    from weft.ir import BoundaryMethod, Segment, SegmentMode

    return Segment(
        segment_id=segment_id, video_id="v", t_start=t_start, t_end=t_end,
        mode=SegmentMode.TRANSCRIPT_ONLY, boundary_method=BoundaryMethod.TOPIC_SHIFT,
        transcript_raw=text, slide_ref=slide_ref,
    )


def test_fingerprint_changes_when_the_segment_covers_different_content():
    """**同一個 segment_id 涵蓋不同內容時，快取不得命中。**

    `segment_id` 是位置編號（`#010`）。換一種分段方式，`#010` 就涵蓋
    完全不同的時間範圍——而舊的冪等鍵只有 `segment_id + prompt_version
    + model`，於是照樣命中。實測 cxrqHABhWOU 的 `#010` 從 72–98 秒變成
    564–593 秒，理解結果卻沿用舊的，溯源 0.929 → 0.071，
    **而每一項機械檢查都是綠的**——每個欄位都在、格式都對，
    只是講的是別的地方的事。與 D20 的圖片錯位同一類。
    """
    from weft.stages.cloud import segment_fingerprint

    old = _segment("v#010", 72.0, 98.0, "胰腺體是長這個樣子這裡面有非常多的胰管")
    new = _segment("v#010", 564.0, 593.0, "胸椎T10到T11所以胰腺癌是很簡單的病")
    assert segment_fingerprint(old) != segment_fingerprint(new)


def test_fingerprint_is_stable_for_the_same_input():
    """輸入沒變就不該重跑——否則每次執行都重算，續跑機制形同虛設。"""
    from weft.stages.cloud import segment_fingerprint

    a = _segment("v#001", 10.0, 40.0, "同樣的內容")
    b = _segment("v#001", 10.0, 40.0, "同樣的內容")
    assert segment_fingerprint(a) == segment_fingerprint(b)


def test_fingerprint_tracks_the_referenced_slide():
    """同樣的話落在不同投影片上時是不同的輸入——`slide_context` 會不一樣。"""
    from weft.stages.cloud import segment_fingerprint

    a = _segment("v#001", 10.0, 40.0, "同樣的話", slide_ref="slide_001")
    b = _segment("v#001", 10.0, 40.0, "同樣的話", slide_ref="slide_002")
    assert segment_fingerprint(a) != segment_fingerprint(b)


def test_cache_without_fingerprint_is_rejected(tmp_path):
    """舊版寫的快取（沒有指紋）**不得命中**。

    讓它命中等於相信一個無法驗證的假設——而那個假設正好就是出事的那個。
    """
    from weft.config import Config
    from weft.ir import Understanding
    from weft.paths import WorkPaths
    from weft.stages.cloud import _load_cached

    cfg = Config()
    work = WorkPaths(tmp_path, "v")
    work.ensure_dirs()
    seg = _segment("v#000", 0.0, 30.0, "內容")
    stale = Understanding(summary="舊的", model_used=cfg.s4.model,
                          prompt_version=cfg.s4.prompt_version)
    work.understanding(0).write_text(stale.model_dump_json(), encoding="utf-8")

    assert _load_cached(work, seg, cfg.s4) is None
