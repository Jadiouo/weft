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
    """批次跑數十支影片時，每支跑完就落地，中途失敗不會前功盡棄。

    **測的是不同影片。** 原本這條拿同一份 IR 寫兩次、斷言行數變兩倍——
    名字說 across videos，內容測的卻是「同一支重跑會複製一份」，
    等於把缺陷寫成期望行為。實測 `out/chunks.jsonl` 因此累積到 428 行，
    相異 id 只有 91 個。
    """
    ir, _, _ = legal_ir
    chunks, _ = build_chunks(ir, cfg())
    path = tmp_path / "chunks.jsonl"
    write_chunks(chunks, path)

    other = ir.model_copy(deep=True)
    other.meta.video_id = "OTHER_VIDEO"
    other_chunks, _ = build_chunks(other, cfg())
    for c in other_chunks:
        c.metadata.video_id = "OTHER_VIDEO"
    write_chunks(other_chunks, path)

    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == (
        len(chunks) + len(other_chunks)
    )


def test_chunks_jsonl_replaces_same_video_on_rerun(legal_ir, tmp_path):
    """重跑同一支影片不得讓 chunks.jsonl 出現重複的 chunk。

    `chunks.jsonl` 是要進向量庫的產品輸出。重複的 chunk 會讓同一段內容
    在檢索結果裡佔掉數個名額。`Chunk.id` 是穩定的，所以重複是可辨識的。
    """
    import json

    ir, _, _ = legal_ir
    chunks, _ = build_chunks(ir, cfg())
    path = tmp_path / "chunks.jsonl"
    write_chunks(chunks, path)
    write_chunks(chunks, path)
    write_chunks(chunks, path)

    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == len(chunks)
    assert len({r["id"] for r in rows}) == len(chunks)



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


# --------------------------------------------------------------------------
# v0.3 首跑發現的溯源漏洞
# --------------------------------------------------------------------------


def test_blocks_citing_an_empty_transcript_are_dropped():
    """逐字稿為空時，`transcript` 來源的 block 無法溯源，應在 S4 就丟棄。

    實測（v0.3 首跑）：片頭 0–50 秒有 12 個 segment 沒有任何字幕。VLM 正確
    判定 `is_slide=false`，卻仍描述畫面上的字並標成 transcript 來源——因為
    prompt 只留了這一個選項給它。9 個這樣的 block 全部溯源失敗，佔未通過
    總數的四分之一。

    這是**設計漏洞不是幻覺**：既沒投影片又沒逐字稿的段落，本來就不該有
    任何 content_block。
    """
    from weft.config import S4Config
    from weft.ir import BoundaryMethod, Segment, SegmentMode
    from weft.stages.understand import to_understanding

    segment = Segment(
        segment_id="v#000", video_id="v", t_start=0.0, t_end=2.0,
        mode=SegmentMode.SPEAKER_ONLY, boundary_method=BoundaryMethod.VIDEO_BOUNDS,
        transcript_raw="", transcript_corrected="",
    )
    raw = {
        "segment_id": "v#000",
        "is_slide": False,
        "reject_reason": "片頭動畫",
        "slide_text": "",
        "corrections": [],
        "summary": "影片片頭",
        "content_blocks": [{
            "type": "口頭延伸",
            "text": "影片開始的片頭動畫。",
            "provenance_kind": "transcript",
            "provenance_ref": "0.0-2.0",
        }],
        "terms": [],
    }
    # v0.4：is_slide 與 reject_reason 由 **S4a** 判定，透過 Slide 傳進來
    from weft.ir import Slide

    slide = Slide(slide_id="slide_001", image_path="03_slides/slide_001.png",
                  t_first_seen=0.0, t_last_seen=2.0, reject_reason="片頭動畫")
    understanding = to_understanding(raw, segment, S4Config(), slide_obj=slide)
    assert understanding.content_blocks == [], "來源為空的 block 應被丟棄"
    assert understanding.is_slide is False
    assert understanding.reject_reason == "片頭動畫"


def test_blocks_with_a_real_transcript_are_kept():
    """有逐字稿時當然要保留——上一條不能誤殺正常情形。"""
    from weft.config import S4Config
    from weft.ir import BoundaryMethod, Segment, SegmentMode
    from weft.stages.understand import to_understanding

    segment = Segment(
        segment_id="v#001", video_id="v", t_start=2.0, t_end=20.0,
        mode=SegmentMode.SPEAKER_ONLY, boundary_method=BoundaryMethod.SLIDE_SWITCH,
        transcript_raw="講者用簽約來比喻識蘊進入的時機。",
        transcript_corrected="講者用簽約來比喻識蘊進入的時機。",
    )
    raw = {
        "segment_id": "v#001", "is_slide": False, "slide_text": "",
        "corrections": [], "summary": "比喻", "terms": [],
        "content_blocks": [{
            "type": "口頭延伸", "text": "講者以簽約作比喻。",
            "provenance_kind": "transcript", "provenance_ref": "2.0-20.0",
        }],
    }
    assert len(to_understanding(raw, segment, S4Config()).content_blocks) == 1


def test_debug_markdown_image_links_resolve_from_the_markdown_location(tmp_path):
    """D23：圖片連結要從 markdown 所在目錄算，不是從 work 目錄。

    §5.6 的人工複核靠看圖——D20 那個 30.6% 的圖片錯位就是這樣抓到的。
    連結壞掉等於這道把關失效，而且不會有任何測試變紅。
    """
    from weft.stages.render import write_debug_markdown
    from tests.factories import make_ir

    work_dir = tmp_path / "work" / "vid"
    ir = make_ir(work_dir)
    slide = ir.slides[0]
    img = work_dir / slide.image_path
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    md_path = tmp_path / "out" / "debug" / "vid.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    write_debug_markdown(ir, work_dir, md_path)

    import re
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md_path.read_text(encoding="utf-8"))
    assert links, "markdown 裡沒有任何圖片連結"
    for link in links:
        assert (md_path.parent / link).exists(), f"連結指不到實際檔案：{link}"


# --------------------------------------------------------------------------
# 產品輸出的重跑語意（票 01／02）
#
# 這幾條在這裡而不在 `test_e2e_offline.py`：它們用 `monkeypatch` 模擬
# 「閘門擋下」與「寫檔失敗」，而 §5.5 #10 規定只有 `_unit_` 檔可以 mock。
# 測的也確實是 `render.py` 的寫檔行為，不是端到端。
# --------------------------------------------------------------------------


@pytest.fixture
def out_paths(tmp_path):
    from weft.paths import OutPaths

    out = OutPaths(tmp_path / "out")
    out.ensure_dirs()
    return out

def test_needs_review_removes_previously_written_chunks(legal_ir, tmp_path, out_paths, monkeypatch):
    """影片被 §5.4 擋下時，**上一版的 chunk 不得留在產品輸出裡**。

    情境是真的會發生的：某支第一次跑通過、chunk 進了 chunks.jsonl；
    之後改了 prompt 或溯源基準（票 01 就是），通過率掉到門檻以下 →
    log 印「不寫入 chunks.jsonl」，但舊的內容原封不動留著。
    那比重複更糟——一支已被判定不可信的影片，繼續用上一版的內容
    留在知識庫裡，而且沒有任何一步會發現。
    """
    import json

    from weft.stages.cloud import s6_render

    from weft.config import Config
    from weft.paths import WorkPaths

    ir, _transcript, base = legal_ir
    cfg = Config()
    cfg.work_dir = tmp_path / "work"
    cfg.out_dir = tmp_path / "out"
    work = WorkPaths(cfg.work_dir, ir.meta.video_id)
    work.ensure_dirs()
    for png in (base / "03_slides").glob("*.png"):
        (work.dir / "03_slides" / png.name).write_bytes(png.read_bytes())
    out = out_paths

    chunks = s6_render(cfg, ir, work, out)
    assert chunks and out.chunks.exists()

    # 讓這支影片這次過不了閘門
    monkeypatch.setattr("weft.validation.thresholds.MAX_UNVERIFIED_RATIO", -1.0)
    monkeypatch.setattr("weft.validation.provenance.MAX_UNVERIFIED_RATIO", -1.0)
    assert s6_render(cfg, ir, work, out) == []

    rows = [json.loads(x) for x in out.chunks.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert not [r for r in rows if r["metadata"]["video_id"] == ir.meta.video_id], (
        "被擋下的影片仍有 chunk 留在 chunks.jsonl"
    )


def test_other_videos_survive_a_blocked_rerun(out_paths):
    """擋下一支影片時不得波及別支。"""
    import json

    from weft.stages.render import drop_video_from_chunks

    out = out_paths
    out.chunks.parent.mkdir(parents=True, exist_ok=True)
    out.chunks.write_text(
        "\n".join(json.dumps({"id": f"{v}#0", "metadata": {"video_id": v}})
                  for v in ("keep_me", "drop_me")) + "\n",
        encoding="utf-8",
    )
    drop_video_from_chunks(out.chunks, "drop_me")

    rows = [json.loads(x) for x in out.chunks.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [r["metadata"]["video_id"] for r in rows] == ["keep_me"]


def test_rewrite_is_atomic(legal_ir, tmp_path, out_paths, monkeypatch):
    """寫到一半失敗時，既有內容不得消失。

    改成「讀出→截斷→重寫」之後，最壞情況從「壞掉最後一行」變成
    「整批不見」——批次跑數十支影片時那是實質的資料遺失。
    """
    import json

    from weft.stages import render

    out = out_paths
    out.chunks.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"id": "keep#0", "metadata": {"video_id": "keep"}}) + "\n"
    out.chunks.write_text(original, encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("磁碟滿了")

    monkeypatch.setattr(render.os, "replace", boom, raising=False)
    with pytest.raises(OSError):
        render.drop_video_from_chunks(out.chunks, "keep")

    assert out.chunks.read_text(encoding="utf-8") == original


# --------------------------------------------------------------------------
# 票 03：per-video 閘門 vs 全域只記錄
# --------------------------------------------------------------------------


def _verdict(video_id: str, total: int, unverified: int, *, wrong=0, dep=0):
    from weft.ir import VerificationStatus
    from weft.validation.provenance import BlockVerdict, VideoVerdict

    rows = []
    for i in range(total):
        bad = i < unverified
        rows.append(BlockVerdict(
            segment_id=f"{video_id}#{i:03d}", block_index=0, content_type="白話解說",
            status=VerificationStatus.UNVERIFIED if bad else VerificationStatus.VERIFIED,
            similarity=0.1 if bad else 0.9, copy_ratio=0.1,
            wrong_source=bad and i < wrong,
            depends_on_correction=bad and wrong <= i < wrong + dep,
        ))
    return VideoVerdict(video_id=video_id, verdicts=rows)


def test_provenance_record_breaks_down_by_cause(tmp_path):
    """逐支記錄要帶**成因分解**，只給一個比率等於沒說。

    R27：27 筆未通過裡 17 筆的病根不在內容品質。三種成因的修法完全不同，
    合成一個數字之後就指不出該修哪裡。
    """
    from weft.stages.render import write_provenance_record

    path = tmp_path / "provenance.jsonl"
    record = write_provenance_record(_verdict("v1", 10, 4, wrong=2, dep=1), path)

    assert record["blocks"] == 10
    assert record["verified"] == 6
    assert record["wrong_source"] == 2
    assert record["depends_on_correction"] == 1
    assert record["unresolved"] == 1
    assert record["wrong_source"] + record["depends_on_correction"] + record["unresolved"] == 4


def test_provenance_record_replaces_same_video(tmp_path):
    """重跑同一支要取代，不是累積——否則趨勢會被自己的歷史污染。"""
    import json

    from weft.stages.render import write_provenance_record

    path = tmp_path / "provenance.jsonl"
    write_provenance_record(_verdict("v1", 10, 5), path)
    write_provenance_record(_verdict("v2", 10, 0), path)
    write_provenance_record(_verdict("v1", 10, 0), path)

    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert sorted(r["video_id"] for r in rows) == ["v1", "v2"]
    assert {r["video_id"]: r["verified"] for r in rows} == {"v1": 10, "v2": 10}


def test_overall_rate_is_recorded_not_gated(tmp_path):
    """全域比率是**趨勢**，不是驗收——它不得出現在 ACCEPTANCE_THRESHOLDS。

    §5.2 原本把 per-video 閘門寫成「對象：全部」的全域門檻，於是四支合計
    的 0.838 被當成驗收依據。那個數字混了四種成因，本來就不該有門檻。
    """
    import weft.validation.thresholds as T
    from weft.stages.render import overall_provenance_rate, write_provenance_record

    path = tmp_path / "provenance.jsonl"
    write_provenance_record(_verdict("v1", 10, 5), path)
    write_provenance_record(_verdict("v2", 10, 1), path)

    overall = overall_provenance_rate(path)
    assert overall["videos"] == 2
    assert overall["blocks"] == 20
    assert overall["provenance_rate_overall"] == 0.7

    assert "provenance_rate_overall" in T.OBSERVED_ONLY
    assert "provenance_rate_overall" not in T.ACCEPTANCE_THRESHOLDS
    assert not any("OVERALL" in n for n in T.ACCEPTANCE_THRESHOLDS), (
        "全域比率不得成為驗收門檻"
    )


def test_gate_is_per_video_not_overall(tmp_path):
    """一支很差不得拖垮另一支，反之亦然。

    這正是把兩種語意合成一個數字時會發生的事：合計 0.70 看起來「沒過」，
    但實際上 v2 是乾淨的、只有 v1 該被擋。
    """
    from weft.stages.render import overall_provenance_rate, write_provenance_record

    path = tmp_path / "provenance.jsonl"
    bad = write_provenance_record(_verdict("v1", 10, 5), path)
    good = write_provenance_record(_verdict("v2", 10, 0), path)

    assert bad["needs_review"] is True
    assert good["needs_review"] is False
    assert overall_provenance_rate(path)["gated_out"] == 1
