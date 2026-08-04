"""S4 的圖片↔區段綁定（D20）。

v0.3 首跑把一整批的文字放前面、圖片全部裸接在後面，只靠一句「依序對應」。
實測 49 個區段中 **15 個（30.6%）拿到隔壁那張圖的分析結果**，且每一個錯都
落在批次內、沒有一個跨批次邊界。量測見 `experiments/r14_image_binding/`。

這些測試釘住修法：**每張圖緊接在它自己的區段標頭之後**，且缺圖時不得
靜靜跳過（跳過會讓後面的圖遞補上來，整批位移）。
"""

from __future__ import annotations

from weft.ir import BoundaryMethod, Segment, SegmentMode
from weft.stages.understand import build_parts


def _seg(n: int, ref: str | None) -> Segment:
    return Segment(
        segment_id=f"vid#{n:03d}",
        video_id="vid",
        t_start=float(n * 10),
        t_end=float(n * 10 + 10),
        mode=SegmentMode.SLIDE if ref else SegmentMode.SPEAKER_ONLY,
        candidate_ref=ref,
        cue_indices=[n],
        transcript_raw=f"第{n}段的逐字稿",
        transcript_corrected=f"第{n}段的逐字稿",
        boundary_method=BoundaryMethod.SLIDE_SWITCH,
    )


def test_each_image_follows_its_own_segment_header():
    segs = [_seg(0, "slide_001"), _seg(1, "slide_002"), _seg(2, "slide_003")]
    parts = build_parts(segs, None)

    # 尾端是總指示，沒有圖
    assert parts[-1][1] is None

    body = parts[:-1]
    assert len(body) == 3
    for seg, (text, image_key) in zip(segs, body, strict=True):
        assert seg.segment_id in text
        assert image_key == seg.candidate_ref, "圖必須綁在自己的區段上"


def test_segment_without_image_does_not_shift_the_others():
    """中間那段沒有代表畫面時，後面兩段的圖**不能**往前遞補。"""
    segs = [_seg(0, "slide_001"), _seg(1, None), _seg(2, "slide_003")]
    body = build_parts(segs, None)[:-1]

    assert [k for _, k in body] == ["slide_001", None, "slide_003"]
    assert "沒有代表畫面" in body[1][0]


def test_prev_summary_is_a_separate_part_with_no_image():
    segs = [_seg(0, "slide_001")]
    parts = build_parts(segs, "前一段講到胎體成長")
    assert parts[0][1] is None
    assert "前一段講到胎體成長" in parts[0][0]
    # 摘要不能吃掉第一段的圖
    assert parts[1][1] == "slide_001"


def test_header_points_at_the_immediately_following_image():
    """措辭必須指向『緊接的那一張』，不是『附件依序對應』。

    首跑的措辭是後者，模型無法可靠地把第 N 張圖對到第 N 段。
    """
    text, _ = build_parts([_seg(0, "slide_001")], None)[0]
    assert "緊接" in text
    assert "依序對應" not in text


def test_tail_instruction_forbids_cross_segment_image_use():
    tail, image_key = build_parts([_seg(0, "slide_001")], None)[-1]
    assert image_key is None
    assert "不要參考其他段的圖" in tail
