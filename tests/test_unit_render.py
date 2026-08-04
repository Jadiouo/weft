"""chunks.jsonl 與 debug markdown。SDD §4.9、§3.5。

chunk 是**產品**——最終進向量庫的就是這些。§3.5 的自足性因此是硬約束：
「`text` 欄位不得包含『如上圖』『前面提到的』等指涉性語句，也不得包含
markdown 裝飾。」

一個含「如上圖」的 chunk 被檢索出來時，讀者看不到任何圖，那條資料就是廢的。
"""

from __future__ import annotations

import json

import pytest

from weft.config import S6Config
from weft.ir import (
    ContentBlock,
    ContentType,
    Provenance,
    ProvenanceKind,
    VerificationStatus,
)
from weft.stages.render import (
    build_chunks,
    referential_phrases,
    split_long_text,
    strip_markdown,
    write_chunks,
    write_debug_markdown,
)
from weft.validation import invariants as inv
from weft.validation.thresholds import MAX_CHUNK_CHARS


def cfg() -> S6Config:
    return S6Config()


# --------------------------------------------------------------------------
# §3.5 自足性
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    ["如上圖", "如下圖", "前面提到", "上述", "這個式子", "這張圖", "剛才說的"],
)
def test_referential_phrases_are_detected(phrase):
    assert referential_phrases(f"{phrase}所說的內容非常重要") == [phrase]


def test_self_contained_text_has_no_referential_phrases():
    assert referential_phrases("一月為胞，是指精血凝聚成形的階段。") == []


def test_markdown_decoration_is_stripped():
    """向量庫存的是純文字，`**` 會變成檢索雜訊。"""
    assert strip_markdown("**一月為胞**，精血凝也") == "一月為胞，精血凝也"
    assert strip_markdown("## 標題\n- 條列項") == "標題 條列項"
    assert strip_markdown("`程式碼`") == "程式碼"


def test_stripping_collapses_whitespace():
    assert strip_markdown("一月為胞\n\n精血凝也") == "一月為胞 精血凝也"


# --------------------------------------------------------------------------
# §4.9 chunk 切分
# --------------------------------------------------------------------------


def test_short_text_is_not_split():
    assert split_long_text("一月為胞，精血凝也。", MAX_CHUNK_CHARS) == ["一月為胞，精血凝也。"]


def test_long_text_splits_at_sentence_boundaries():
    """切在句中會產生讀不通的 chunk，而 chunk 是要單獨被檢索出來給人看的。"""
    sentence = "一月為胞精血凝也。"
    pieces = split_long_text(sentence * 200, 100)

    assert len(pieces) > 1
    for piece in pieces:
        assert piece.endswith("。"), f"切在句中：{piece[-20:]!r}"
        assert len(piece) <= 100


def test_split_preserves_all_content():
    """切分不得漏字。"""
    text = "一月為胞，精血凝也。二月為胎，形兆胚也。三月陽神為三魂，動而生也。" * 30
    assert "".join(split_long_text(text, 100)) == text


def test_unsplittable_long_text_is_hard_split_not_dropped():
    """沒有標點的超長段落只能硬切——但不得整段丟掉。"""
    text = "一" * 500
    pieces = split_long_text(text, 100)
    assert "".join(pieces) == text
    assert all(len(p) <= 100 for p in pieces)


def test_max_chunk_chars_matches_sdd():
    assert MAX_CHUNK_CHARS == 800
    assert cfg().max_chunk_chars == MAX_CHUNK_CHARS


# --------------------------------------------------------------------------
# chunk 建構
# --------------------------------------------------------------------------


def test_one_content_block_becomes_one_chunk(legal_ir):
    """SDD §4.9 的切分規則。"""
    ir, _, _ = legal_ir
    chunks, _ = build_chunks(ir, cfg())
    assert len(chunks) == len(ir.segments[1].understanding.content_blocks)


def test_chunk_metadata_is_complete(legal_ir):
    """§5.3 不變量 8。"""
    ir, _, _ = legal_ir
    chunks, _ = build_chunks(ir, cfg())
    assert inv.rule_08_chunk_metadata_complete(chunks) == []


def test_chunk_url_carries_the_timestamp(legal_ir):
    """§3.5：url 要能跳回影片的具體秒數。這是「可溯源」的最後一哩。"""
    ir, _, _ = legal_ir
    chunks, _ = build_chunks(ir, cfg())
    seg = ir.segments[1]
    assert chunks[0].metadata.url.endswith(f"&t={int(seg.t_start)}s")


def test_chunk_ids_are_unique(legal_ir):
    ir, _, _ = legal_ir
    chunks, _ = build_chunks(ir, cfg())
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_split_chunks_each_carry_full_metadata(legal_ir):
    """§4.9：「每個切片都要複製完整 metadata」。"""
    ir, _, _ = legal_ir
    block = ir.segments[1].understanding.content_blocks[0]
    block.text = "一月為胞，精血凝也。" * 200

    chunks, _ = build_chunks(ir, cfg())
    split = [c for c in chunks if c.id.startswith(f"{ir.segments[1].segment_id}#b00")]

    assert len(split) > 1
    assert inv.rule_08_chunk_metadata_complete(split) == []
    assert len({c.metadata.video_id for c in split}) == 1


def test_segments_without_understanding_produce_no_chunks(legal_ir):
    """§4.7 失敗行為留下的 understanding=null 不該變成空 chunk。"""
    ir, _, _ = legal_ir
    ir.segments[1].understanding = None
    chunks, _ = build_chunks(ir, cfg())
    assert chunks == []


def test_referential_text_is_flagged(legal_ir):
    """S4 的 prompt 已要求展開指涉語句，這裡是最後一道檢查——
    漏網的必須被記錄，不能靜默進入產品。"""
    ir, _, _ = legal_ir
    ir.segments[1].understanding.content_blocks[0].text = "如上圖所示，一月為胞。"
    _chunks, warnings = build_chunks(ir, cfg())
    assert any("如上圖" in w for w in warnings)


def test_markdown_is_stripped_from_chunk_text(legal_ir):
    ir, _, _ = legal_ir
    ir.segments[1].understanding.content_blocks[0].text = "**一月為胞**，精血凝也。"
    chunks, _ = build_chunks(ir, cfg())
    assert "**" not in chunks[0].text


# --------------------------------------------------------------------------
# §5.4 溯源閘門
# --------------------------------------------------------------------------


def test_unverified_blocks_are_excluded_from_chunks(legal_ir):
    """§5.4：溯源未通過的內容不得進入產品。"""
    ir, _, _ = legal_ir
    blocks = ir.segments[1].understanding.content_blocks
    blocks[0].verification = VerificationStatus.UNVERIFIED
    blocks[1].verification = VerificationStatus.VERIFIED

    chunks, warnings = build_chunks(ir, cfg())
    assert len(chunks) == 1
    assert any("溯源" in w for w in warnings)


def test_degenerate_copy_blocks_are_excluded(legal_ir):
    """§5.5 #5：不得為了通過溯源檢查而讓 LLM 只做複製貼上。"""
    ir, _, _ = legal_ir
    for block in ir.segments[1].understanding.content_blocks:
        block.verification = VerificationStatus.DEGENERATE_COPY
    chunks, _ = build_chunks(ir, cfg())
    assert chunks == []


def test_blocks_without_verdict_are_kept(legal_ir):
    """尚未跑過溯源檢查（verification is None）不等於未通過。"""
    ir, _, _ = legal_ir
    chunks, _ = build_chunks(ir, cfg())
    assert len(chunks) == 2


# --------------------------------------------------------------------------
# 輸出
# --------------------------------------------------------------------------


def test_chunks_jsonl_is_one_object_per_line(legal_ir, tmp_path):
    ir, _, _ = legal_ir
    chunks, _ = build_chunks(ir, cfg())
    path = tmp_path / "chunks.jsonl"
    write_chunks(chunks, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(chunks)
    for line in lines:
        json.loads(line)  # 每行都必須是合法 JSON


def test_chunks_jsonl_appends_across_videos(legal_ir, tmp_path):
    """批次跑數十支影片時，每支跑完就落地，中途失敗不會前功盡棄。"""
    ir, _, _ = legal_ir
    chunks, _ = build_chunks(ir, cfg())
    path = tmp_path / "chunks.jsonl"
    write_chunks(chunks, path)
    write_chunks(chunks, path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2 * len(chunks)


def test_debug_markdown_contains_clickable_timestamps(legal_ir, tmp_path):
    """§4.9：debug markdown 須含內嵌圖片與可點的時間戳連結（§5.6 人工抽檢用）。"""
    ir, _, base = legal_ir
    path = write_debug_markdown(ir, base, tmp_path / "debug.md")
    body = path.read_text(encoding="utf-8")

    assert "&t=" in body
    assert "03_slides/slide_001.png" in body
    assert ir.meta.title in body


def test_debug_markdown_shows_term_corrections(legal_ir, tmp_path):
    """§5.6 的檢查項目之一：逐字稿術語是否正確。"""
    ir, _, base = legal_ir
    body = write_debug_markdown(ir, base, tmp_path / "d.md").read_text(encoding="utf-8")
    assert "時運" in body and "識蘊" in body


def test_debug_markdown_flags_unverified_blocks(legal_ir, tmp_path):
    ir, _, base = legal_ir
    ir.segments[1].understanding.content_blocks[0].verification = VerificationStatus.UNVERIFIED
    body = write_debug_markdown(ir, base, tmp_path / "d.md").read_text(encoding="utf-8")
    assert "unverified" in body


def test_debug_markdown_notes_progressive_merge(legal_ir, tmp_path):
    """§5.6 的檢查項目：投影片圖是否為該頁最完整版本。"""
    ir, _, base = legal_ir
    body = write_debug_markdown(ir, base, tmp_path / "d.md").read_text(encoding="utf-8")
    assert "逐條動畫" in body
