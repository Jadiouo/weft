"""SDD §5.3 不變量檢查器自身的測試。

檢查器是整個驗證框架的骨幹。一個「永遠回傳空清單」的檢查器會讓所有測試
變綠，卻什麼都沒保證——所以每條規則都要有反例，證明它真的會抓到東西。

作法：從合法 IR 出發，只破壞一個性質，斷言對應規則被觸發。
"""

from __future__ import annotations

import copy

import pytest

from weft.ir import (
    Chunk,
    ChunkMetadata,
    ContentType,
    Provenance,
    ProvenanceKind,
    Transcript,
)
from weft.validation import invariants as inv


def rules_hit(violations) -> set[int]:
    return {v.rule for v in violations}


# --------------------------------------------------------------------------
# 正例
# --------------------------------------------------------------------------


def test_legal_ir_passes_all_invariants(legal_ir):
    ir, transcript, base = legal_ir
    assert inv.check_all(ir, transcript, base) == []


def test_assert_all_does_not_raise_on_legal_ir(legal_ir):
    ir, transcript, base = legal_ir
    inv.assert_all(ir, transcript, base)  # 不應 raise


def test_legal_chunks_pass(legal_ir):
    ir, _, _ = legal_ir
    chunks = make_chunks(ir)
    assert inv.rule_08_chunk_metadata_complete(chunks) == []


def make_chunks(ir) -> list[Chunk]:
    seg = ir.segments[1]
    block = seg.understanding.content_blocks[0]
    return [
        Chunk(
            id=f"{seg.segment_id}#b00",
            text=block.text,
            metadata=ChunkMetadata(
                video_id=ir.meta.video_id,
                series_id=ir.meta.series_id,
                video_title=ir.meta.title,
                episode_index=ir.meta.episode_index,
                t_start=seg.t_start,
                t_end=seg.t_end,
                url=f"{ir.meta.url}&t={int(seg.t_start)}s",
                content_type=block.type,
                slide_ref=seg.slide_ref,
                terms=seg.understanding.terms,
                provenance_kind=block.provenance.kind,
            ),
        )
    ]


# --------------------------------------------------------------------------
# 反例：每條規則各自可被觸發
# --------------------------------------------------------------------------


def test_rule_01_detects_overlap(legal_ir):
    ir, _, _ = legal_ir
    ir.segments[1].t_start = 10.0  # 侵入前一段 [0, 20]
    assert 1 in rules_hit(inv.rule_01_segments_disjoint(ir))


def test_rule_02_detects_gap(legal_ir):
    ir, _, _ = legal_ir
    ir.segments[0].t_end = 5.0  # 與下一段之間留下 15 秒空隙
    assert 2 in rules_hit(inv.rule_02_segments_cover_video(ir))


def test_rule_02_detects_uncovered_tail(legal_ir):
    ir, _, _ = legal_ir
    ir.segments[-1].t_end = 60.0  # 影片全長 90
    assert 2 in rules_hit(inv.rule_02_segments_cover_video(ir))


def test_rule_02_tolerates_sub_second_gap(legal_ir):
    """容忍 ±1 秒是 SDD 明文，不是可以再放寬的模糊地帶。"""
    ir, _, _ = legal_ir
    ir.segments[0].t_end = 19.4
    assert inv.rule_02_segments_cover_video(ir) == []


def test_rule_03_detects_unassigned_cue(legal_ir):
    ir, transcript, _ = legal_ir
    ir.segments[1].cue_indices = [1, 2]  # cue 3 沒人認領
    assert 3 in rules_hit(inv.rule_03_cues_assigned_once(ir, transcript))


def test_rule_03_detects_double_assignment(legal_ir):
    ir, transcript, _ = legal_ir
    ir.segments[0].cue_indices = [0, 1]  # cue 1 被兩段同時認領
    assert 3 in rules_hit(inv.rule_03_cues_assigned_once(ir, transcript))


def test_rule_03_detects_phantom_cue(legal_ir):
    ir, transcript, _ = legal_ir
    ir.segments[1].cue_indices = [1, 2, 3, 99]
    assert 3 in rules_hit(inv.rule_03_cues_assigned_once(ir, transcript))


def test_rule_04_detects_dangling_slide_ref(legal_ir):
    ir, _, _ = legal_ir
    ir.segments[1].slide_ref = "slide_999"
    assert 4 in rules_hit(inv.rule_04_slide_refs_exist(ir))


def test_rule_04_detects_slide_mode_without_ref(legal_ir):
    ir, _, _ = legal_ir
    ir.segments[1].slide_ref = None
    assert 4 in rules_hit(inv.rule_04_slide_refs_exist(ir))


def test_rule_04_detects_dangling_provenance_ref(legal_ir):
    ir, _, _ = legal_ir
    ir.segments[1].understanding.content_blocks[0].provenance = Provenance(
        kind=ProvenanceKind.SLIDE_OCR, ref="slide_404"
    )
    assert 4 in rules_hit(inv.rule_04_slide_refs_exist(ir))


def test_rule_05_detects_missing_image(legal_ir):
    ir, _, base = legal_ir
    (base / "03_slides" / "slide_001.png").unlink()
    assert 5 in rules_hit(inv.rule_05_images_exist(ir, base))


def test_rule_05_detects_corrupt_image(legal_ir):
    """「存在」不等於「可開啟」——半個 PNG 也會存在。"""
    ir, _, base = legal_ir
    (base / "03_slides" / "slide_001.png").write_bytes(b"\x89PNG\r\n\x1a\n truncated")
    assert 5 in rules_hit(inv.rule_05_images_exist(ir, base))


def test_rule_06_detects_out_of_range_timestamp(legal_ir):
    ir, transcript, _ = legal_ir
    ir.segments[1].t_end = 9999.0
    assert 6 in rules_hit(inv.rule_06_timestamps_sane(ir, transcript))


def test_rule_06_detects_inverted_segment(legal_ir):
    ir, transcript, _ = legal_ir
    ir.segments[1].t_start = 80.0
    ir.segments[1].t_end = 30.0
    assert 6 in rules_hit(inv.rule_06_timestamps_sane(ir, transcript))


def test_rule_06_detects_unsorted_segments(legal_ir):
    ir, transcript, _ = legal_ir
    ir.segments.reverse()
    assert 6 in rules_hit(inv.rule_06_timestamps_sane(ir, transcript))


def test_rule_07_detects_blank_provenance(legal_ir):
    ir, _, _ = legal_ir
    ir.segments[1].understanding.content_blocks[0].provenance.ref = "   "
    assert 7 in rules_hit(inv.rule_07_provenance_not_null(ir))


def test_rule_07_provenance_cannot_be_omitted_at_construction():
    """schema 層就該擋掉——§5.4 說 provenance 是必填欄位。"""
    from pydantic import ValidationError

    from weft.ir import ContentBlock

    with pytest.raises(ValidationError):
        ContentBlock(type=ContentType.SCRIPTURE, text="一月為胞")


def test_rule_08_detects_null_metadata_field(legal_ir):
    ir, _, _ = legal_ir
    chunks = make_chunks(ir)
    chunks[0].metadata.video_title = ""
    assert 8 in rules_hit(inv.rule_08_chunk_metadata_complete(chunks))


def test_rule_08_allows_v2_reserved_nulls(legal_ir):
    """series_id / episode_index 是 §7.5 的 v2 預留欄位，單支影片來源時本就無值。"""
    ir, _, _ = legal_ir
    chunks = make_chunks(ir)
    chunks[0].metadata.series_id = None
    chunks[0].metadata.episode_index = None
    assert inv.rule_08_chunk_metadata_complete(chunks) == []


def test_rule_08_requires_slide_ref_when_provenance_is_slide(legal_ir):
    ir, _, _ = legal_ir
    chunks = make_chunks(ir)
    chunks[0].metadata.slide_ref = None  # 但 provenance_kind 仍是 slide_ocr
    assert 8 in rules_hit(inv.rule_08_chunk_metadata_complete(chunks))


def test_rule_08_allows_missing_slide_ref_for_transcript_provenance(legal_ir):
    ir, _, _ = legal_ir
    chunks = make_chunks(ir)
    chunks[0].metadata.slide_ref = None
    chunks[0].metadata.provenance_kind = ProvenanceKind.TRANSCRIPT
    assert inv.rule_08_chunk_metadata_complete(chunks) == []


def test_rule_09_detects_tampered_raw_text(legal_ir):
    """這條擋的是「某階段偷偷改寫 transcript_raw」——§4.5 明文禁止。"""
    _, transcript, _ = legal_ir
    transcript.cues[0].text_raw = "被某個階段改寫過的內容"
    assert 9 in rules_hit(inv.rule_09_transcript_raw_intact(transcript))


def test_rule_09_detects_shifted_timeline(legal_ir):
    """時間軸被整體平移，文字沒動——一樣算 raw 被改。"""
    _, transcript, _ = legal_ir
    for cue in transcript.cues:
        cue.t_start += 0.5
    assert 9 in rules_hit(inv.rule_09_transcript_raw_intact(transcript))


def test_rule_09_ignores_corrected_field(legal_ir):
    """S2c 就是要寫 text_corrected，寫了不該讓不變量 9 紅燈。"""
    _, transcript, _ = legal_ir
    transcript.cues[0].text_corrected = "改過的內容"
    assert inv.rule_09_transcript_raw_intact(transcript) == []


def test_rule_10_detects_untraceable_correction(legal_ir):
    from weft.ir import Correction

    ir, transcript, _ = legal_ir
    transcript.cues[2].corrections = [
        Correction(**{"from": "根本沒出現過的詞", "to": "識蘊", "source": "slide_001",
                      "method": "lexicon", "score": 0.9})
    ]
    assert 10 in rules_hit(inv.rule_10_corrections_traceable(ir, transcript))


# --------------------------------------------------------------------------
# 執行入口的行為
# --------------------------------------------------------------------------


def test_assert_all_raises_and_lists_every_violation(legal_ir):
    ir, transcript, base = legal_ir
    ir.segments[1].slide_ref = "slide_999"
    transcript.cues[0].text_raw = "竄改"

    with pytest.raises(inv.InvariantViolation) as excinfo:
        inv.assert_all(ir, transcript, base)

    hit = {v.rule for v in excinfo.value.violations}
    assert {4, 9} <= hit
    assert "不變量" in str(excinfo.value)


def test_check_all_is_side_effect_free(legal_ir):
    """檢查器不得就地修改 IR——驗證動作改變被驗證的東西，結果就不可信。"""
    ir, transcript, base = legal_ir
    before = ir.model_dump_json()
    inv.check_all(ir, transcript, base, chunks=make_chunks(ir))
    assert ir.model_dump_json() == before


def test_no_warn_only_escape_hatch():
    """§5.5 #9：不得把 assert 改成 warning 或 log。

    這條測試是機械式的護欄——若有人替 check_all/assert_all 加上
    warn_only / strict=False 之類的參數，這裡就會紅燈。
    """
    import inspect

    for fn in (inv.check_all, inv.assert_all):
        params = set(inspect.signature(fn).parameters)
        forbidden = {"warn_only", "strict", "raise_on_error", "soft", "ignore"}
        assert not (params & forbidden), f"{fn.__name__} 出現了繞過用的參數"
