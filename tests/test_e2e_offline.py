"""離線端到端：不需要網路、不需要 ollama、不需要額度。

**這個檔案回答的問題是「擱置數週後回來，現在還是好的嗎」。**
它必須在一台沒有網路的機器上、在幾秒內跑完。

與 `test_e2e_pipeline.py` 的分工：

| | 依賴 | 涵蓋 |
|---|---|---|
| 這裡 | 無 | S6 渲染、§5.4 溯源、§5.3 不變量、chunks.jsonl 契約 |
| `@pytest.mark.synth` | ffmpeg | S1b–S3 本地管線 |
| `@pytest.mark.live` | 網路 + ollama | S0–S6 全程，實際呼叫模型 |

**這裡沒有 mock 模型**（§5.5 #10）——它根本不呼叫模型。
S4 的產物由 fixture 提供，測的是**拿到理解結果之後**那一段。
那正是擱置後最容易壞掉、又最不需要外部資源就能驗證的部分。

已知的涵蓋缺口，刻意留著並寫在這裡而不是假裝沒有：
fixture 是手寫的 happy path，不是真實影片的 IR。真實資料的形狀
（空 slide_text、批次少回、簡繁混雜）由 `@pytest.mark.live` 那層覆蓋。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from weft.config import Config
from weft.paths import OutPaths, WorkPaths
from weft.validation import invariants as inv

pytestmark = pytest.mark.e2e


@pytest.fixture
def offline_run(tmp_path: Path):
    """把一份合法 IR 擺成 work/ 佈局，回傳 (cfg, ir, transcript, work, out)。

    刻意用 `factories.make_ir` 而不是版控一份真實影片的 IR：
    真實素材是他人著作（SDD §9），而 §3.1 的 work/ 本來就不進版控。
    """
    from tests.factories import make_ir, make_transcript

    cfg = Config()
    cfg.work_dir = tmp_path / "work"
    cfg.out_dir = tmp_path / "out"

    ir = make_ir(tmp_path)
    transcript = make_transcript()

    work = WorkPaths(cfg.work_dir, ir.meta.video_id)
    work.ensure_dirs()
    # `make_ir` 把圖寫在 tmp_path 下，IR 的 image_path 是相對於 work.dir 的
    for src in (tmp_path / "03_slides").glob("*.png"):
        (work.dir / "03_slides" / src.name).write_bytes(src.read_bytes())

    out = OutPaths(cfg.out_dir)
    out.ensure_dirs()
    return cfg, ir, transcript, work, out


def test_render_produces_chunks_without_network_or_models(offline_run):
    """S6 從快取的 IR 產出 chunks.jsonl，全程不碰網路與模型。"""
    from weft.ir import Chunk
    from weft.stages.cloud import s6_render

    cfg, ir, _transcript, work, out = offline_run
    chunks = s6_render(cfg, ir, work, out)

    assert chunks, "S6 沒有產出任何 chunk"
    assert out.chunks.exists()
    written = [
        Chunk.model_validate_json(line)
        for line in out.chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(written) == len(chunks)


def test_chunks_satisfy_all_invariants(offline_run):
    """§5.3 的不變量對真正落地的 chunk 全數成立。

    這是 `chunks.jsonl` 的契約：metadata 完整、時間軸單調、
    圖檔存在、`text_raw` 未被竄改。擱置後最可能悄悄壞掉的就是這一層。
    """
    from weft.stages.cloud import s6_render

    cfg, ir, transcript, work, out = offline_run
    chunks = s6_render(cfg, ir, work, out)

    violations = inv.check_all(ir, transcript, work.dir, chunks=chunks)
    assert violations == [], "\n".join(str(v) for v in violations)


def test_every_chunk_in_the_file_passed_provenance(offline_run):
    """**`chunks.jsonl` 的契約**：檔案裡不存在溯源未通過的內容。

    實測成立（票 13 逐支核對：23/23、46/46、44/44、53/53 對上
    `provenance.jsonl` 的 `verified`），但那時它**只存在於程式碼裡**——
    S6 的一個 `continue`。下游（vault ingest）要據此決定「不必自己再判斷
    可信度」，那就必須是**被釘住的契約**，不是實作的巧合。

    兩層過濾：未通過的 block 在 S6 被排除（§5.4）；
    未過 per-video 閘門的影片整支不寫進檔案。這裡測第一層。
    """
    from weft.ir import VerificationStatus
    from weft.stages.cloud import s6_render

    cfg, ir, _transcript, work, out = offline_run
    chunks = s6_render(cfg, ir, work, out)

    verified_texts = {
        block.text
        for seg in ir.segments if seg.understanding
        for block in seg.understanding.content_blocks
        if block.verification is VerificationStatus.VERIFIED
    }
    rejected = {
        block.text
        for seg in ir.segments if seg.understanding
        for block in seg.understanding.content_blocks
        if block.verification is not None
        and block.verification is not VerificationStatus.VERIFIED
    }
    assert verified_texts, "fixture 裡沒有任何通過溯源的 block，這條測不到東西"

    # chunk 的 text 可能被 §4.9 切過，所以比對「是不是某個被拒 block 的開頭」
    for chunk in chunks:
        for bad in rejected:
            assert not bad.startswith(chunk.text[:40]), (
                f"{chunk.id} 的內容來自溯源未通過的 block——"
                f"§5.4 說那些不得進入產品"
            )


def test_content_sha_lets_downstream_notice_that_a_stable_id_changed(offline_run):
    """`id` 是**位置編號**（`<video>#<段序號>#b<塊序號>`）。

    改 `block_chars`、換分段方法、換一版 S4c prompt——`#010` 還是 `#010`，
    但指的時間範圍與內容整個換掉。這正是 D32：當時位置性讓 S4c 讀到
    別的時間範圍的快取，而**所有機械檢查都是綠的**。D32 修的是快取鍵，
    **匯出的 id 沒修**，所以同一個陷阱會跟著 chunk 進到 vault。

    這裡不改 id 的形狀（要改成什麼得下游先決定用什麼當識別），
    只保證**變了看得出來**。
    """
    from weft.stages.cloud import s6_render
    from weft.stages.render import content_sha

    cfg, ir, _transcript, work, out = offline_run
    chunks = s6_render(cfg, ir, work, out)

    for chunk in chunks:
        assert chunk.metadata.content_sha == content_sha(chunk.text)

    # 同一個 id 換掉內容 → sha 必須不同，否則這個欄位沒有用
    victim = chunks[0]
    assert content_sha(victim.text + "改過") != victim.metadata.content_sha


def test_provenance_runs_and_fills_verdicts(offline_run):
    """§5.4 溯源檢查會就地填回每個 block 的判定與成因。

    只斷言「有跑、有填」，不斷言通過率——那個數字取決於 fixture 內容，
    寫死它會變成測 fixture 而不是測機制。
    """
    from weft.stages.cloud import s6_render
    from weft.validation.provenance import check_video

    cfg, ir, _transcript, work, out = offline_run
    s6_render(cfg, ir, work, out)

    blocks = [b for s in ir.segments if s.understanding for b in s.understanding.content_blocks]
    assert blocks
    assert all(b.verification is not None for b in blocks)
    assert all(b.similarity is not None for b in blocks)

    verdict = check_video(ir, cfg.provenance)
    assert verdict.total == len(blocks)


def test_debug_markdown_image_links_resolve(offline_run):
    """§5.6 的人工複核文件，圖片連結必須真的打得開。

    D23：這些連結曾經**一張都打不開**，而 §5.6 的人工抽檢正是靠看圖
    抓到 D20 的。連結壞掉不會有任何機械檢查報錯。
    """
    import re

    from weft.stages.cloud import s6_render

    cfg, ir, _transcript, work, out = offline_run
    s6_render(cfg, ir, work, out)

    md_path = out.debug_md(ir.meta.video_id)
    assert md_path.exists()
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md_path.read_text(encoding="utf-8"))
    assert links, "複核文件裡沒有任何圖片連結"
    for link in links:
        assert (md_path.parent / link).resolve().exists(), f"圖片連結打不開：{link}"


def test_rerun_is_idempotent(offline_run):
    """同一支影片跑兩次，產出不得變多。

    §6.3 的冪等性在這一層的意思是「重跑不會複製產出」。實測 `chunks.jsonl`
    曾累積到 428 行而相異 id 只有 91 個——重複的 chunk 會讓同一段內容
    在檢索結果裡佔掉數個名額。
    """
    from weft.stages.cloud import s6_render

    cfg, ir, _transcript, work, out = offline_run
    s6_render(cfg, ir, work, out)
    first = out.chunks.read_text(encoding="utf-8")
    s6_render(cfg, ir, work, out)
    assert out.chunks.read_text(encoding="utf-8") == first


def test_no_module_reaches_the_network_at_import_time():
    """匯入 weft 的任何模組都不得需要網路或 API key。

    沒有這一條，「離線可跑」會在某次有人把 client 建構搬到模組層級時
    悄悄失效，而失敗訊息會長得像別的問題。
    """
    import importlib
    import pkgutil

    import weft

    failed = []
    for mod in pkgutil.walk_packages(weft.__path__, prefix="weft."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{mod.name}: {exc}")
    assert not failed, "\n".join(failed)


# --------------------------------------------------------------------------
# 票 09：完全沒有投影片時，管線照樣跑得完
# --------------------------------------------------------------------------


@pytest.fixture
def no_slides_run(tmp_path: Path):
    """一支**完全沒有投影片**的影片：純逐字稿主幹。

    對應訪談與口播素材。v0.5 之前這種影片架構上跑不了——分段的依據是
    投影片切換，沒有投影片就沒有東西可以驅動分段。
    """
    from weft.ir import (
        BoundaryMethod,
        ContentBlock,
        ContentType,
        Provenance,
        ProvenanceKind,
        Segment,
        SegmentMode,
        Understanding,
        VideoIR,
        VideoMeta,
    )

    from tests.factories import make_transcript

    cfg = Config()
    cfg.work_dir = tmp_path / "work"
    cfg.out_dir = tmp_path / "out"

    transcript = make_transcript()
    text = "".join(c.text_raw for c in transcript.cues)
    seg = Segment(
        segment_id="NOSLIDE#000", video_id="NOSLIDE", t_start=0.0, t_end=90.0,
        mode=SegmentMode.TRANSCRIPT_ONLY, boundary_method=BoundaryMethod.TOPIC_SHIFT,
        cue_indices=[c.index for c in transcript.cues],
        transcript_raw=text, transcript_corrected=text,
        understanding=Understanding(
            is_slide=False, summary="訪談片段的摘要。",
            content_blocks=[ContentBlock(
                type=ContentType.ORAL,
                text="講者以簽約作比喻，說明識蘊進入的時機。",
                provenance=Provenance(kind=ProvenanceKind.TRANSCRIPT, ref="0.0s-90.0s"),
            )],
        ),
    )
    ir = VideoIR(
        meta=VideoMeta(video_id="NOSLIDE", title="訪談", duration=90.0,
                       url="https://example.invalid/NOSLIDE"),
        slides=[],  # ← 這裡是重點
        segments=[seg],
    )
    work = WorkPaths(cfg.work_dir, "NOSLIDE")
    work.ensure_dirs()
    out = OutPaths(cfg.out_dir)
    out.ensure_dirs()
    return cfg, ir, transcript, work, out


def test_pipeline_completes_without_any_slides(no_slides_run):
    """沒有投影片時 S6 照樣產出 chunk，不報錯、不中止。"""
    from weft.stages.cloud import s6_render

    cfg, ir, _t, work, out = no_slides_run
    chunks = s6_render(cfg, ir, work, out)
    assert chunks, "無投影片的影片一個 chunk 都沒產出"
    assert out.chunks.exists()


def test_no_slide_ocr_provenance_without_slides(no_slides_run):
    """沒有投影片時不得出現 `slide_ocr` 型的來源——那會是憑空的出處。"""
    from weft.ir import ProvenanceKind
    from weft.stages.cloud import s6_render

    cfg, ir, _t, work, out = no_slides_run
    chunks = s6_render(cfg, ir, work, out)
    kinds = {c.metadata.provenance_kind for c in chunks}
    assert ProvenanceKind.SLIDE_OCR not in kinds
    assert all(c.metadata.slide_ref is None for c in chunks)


def test_all_invariants_hold_without_slides(no_slides_run):
    """§5.3 的十條不變量在無投影片路徑上全數通過。

    第 4、5 條是對 slide 的斷言——它們必須在沒有 slide 時**自然成立**，
    而不是靠呼叫端記得跳過。
    """
    from weft.stages.cloud import s6_render

    cfg, ir, transcript, work, out = no_slides_run
    chunks = s6_render(cfg, ir, work, out)
    violations = inv.check_all(ir, transcript, work.dir, chunks=chunks)
    assert violations == [], "\n".join(str(v) for v in violations)


def test_segmentation_works_on_transcript_alone(no_slides_run):
    """分段完全不需要畫面——這正是 v0.5 讓無投影片素材可行的原因。"""
    from weft.stages.align import topic_windows
    from weft.stages.segment import topic_boundaries

    _cfg, _ir, transcript, _work, _out = no_slides_run
    cuts = topic_boundaries(transcript.cues, block_chars=20, window=2)
    windows = topic_windows(cuts, duration=90.0, candidates=[])
    assert windows
    assert all(w.slide_id is None for w in windows)
    assert windows[0].t_start == 0.0 and windows[-1].t_end == 90.0


def test_slide_stages_are_noops_without_slides(tmp_path):
    """零投影片時 S4a／S4b **靜靜地什麼都不做**，不報錯也不做白工。

    這一條看起來多餘，但它守的是「可選路徑真的可選」——只要哪一天有人
    在那些函式裡加了 `slides[0]` 或 `assert slides`，無投影片的素材就會
    在跑到一半的時候炸掉，而那時已經花掉了下載與抽幀的時間。
    """
    from weft.stages.lexicon import apply_to_video, build_lexicon
    from weft.stages.slides import s4a_understand_slides

    cfg = Config()
    work = WorkPaths(tmp_path, "NOSLIDE")
    work.ensure_dirs()

    stats = s4a_understand_slides(cfg, work, [])
    assert stats["representatives"] == 0
    assert stats["failed"] == 0

    assert build_lexicon([]) == set()
    assert apply_to_video([], [], None, cfg.s4b) == 0
