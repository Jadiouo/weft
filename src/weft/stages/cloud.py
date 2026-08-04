"""S4–S6：理解、統整、渲染。SDD §4.7–§4.9，屬 Phase 2–3（§7.3–§7.4）。

S4 是**唯一花額度的階段**。額度來源固定為 Gemini API key（AI Studio free
tier）；§5.5 #12 明文禁止把 AI Pro/Ultra 的訂閱額度接入程式化呼叫。
"""

from __future__ import annotations

import logging

from ..config import Config
from ..ir import Chunk, Lexicon, Segment, Understanding, VideoIR
from ..paths import OutPaths, WorkPaths
from . import pending

log = logging.getLogger(__name__)


def _batches(segments: list[Segment], size: int) -> list[list[Segment]]:
    """把相鄰 segment 分組。SDD §4.7：可將 2–3 個合併為一次呼叫節省額度。"""
    size = max(1, size)
    return [segments[i : i + size] for i in range(0, len(segments), size)]


def s4_understand(
    cfg: Config,
    work: WorkPaths,
    segments: list[Segment],
    lexicon: Lexicon | None,
) -> list[Understanding]:
    """S4 聯合理解。SDD §4.7。

    冪等鍵：segment_id + prompt_version + model
    失敗行為：單一 segment 失敗 → 重試 2 次（指數退避）→ understanding=null，繼續
    """
    import json

    from ..quota import QuotaExhausted, QuotaLedger
    from .understand import ApiKeyMissing, call_gemini, to_understanding, with_retries

    p = cfg.s4
    ledger = QuotaLedger(OutPaths(cfg.out_dir).quota_db, cfg.quota)
    work.understanding_dir.mkdir(parents=True, exist_ok=True)

    slides_by_id: dict[str, str] = {}
    if work.ocr.exists():
        for row in json.loads(work.ocr.read_text(encoding="utf-8")):
            slides_by_id[row["slide_id"]] = row.get("ocr_text") or ""
    image_paths = {sid: work.slides_dir / f"{sid}.png" for sid in slides_by_id}

    terms = [e.term for e in (lexicon.entries if lexicon else [])][:60]

    results: list[Understanding] = []
    prev_summary: str | None = None

    for batch in _batches(segments, p.batch_segments):
        # 續跑：整批都已有結果就直接讀檔，不重複花額度（§6.3）
        cached = [_load_cached(work, seg, p) for seg in batch]
        if all(c is not None for c in cached):
            results += [c for c in cached if c is not None]
            prev_summary = results[-1].summary if results else prev_summary
            continue

        # §6.1 主動節流：呼叫**前**估算，不靠撞 429（§5.5 #13）
        try:
            ledger.check(planned_requests=1, model=p.model)
        except QuotaExhausted as exc:
            log.warning("%s；停止本日處理，進度已保存（§6.1）", exc)
            break

        try:
            batch_result = with_retries(
                lambda b=batch: call_gemini(
                    b, slides_by_id, image_paths, prev_summary, terms, p
                ),
                p.max_retries,
                p.retry_backoff_sec,
            )
        except ApiKeyMissing:
            raise
        except Exception as exc:  # noqa: BLE001
            # §4.7 失敗行為：重試後仍失敗 → understanding=null，繼續下一批
            log.error("批次 %s 失敗，標記為 null 並繼續：%s",
                      [s.segment_id for s in batch], exc)
            ledger.record(p.model, 0, 0, batch[0].segment_id, "error")
            for seg in batch:
                seg.understanding = None
            continue

        ledger.record(
            p.model, batch_result.input_tokens, batch_result.output_tokens,
            batch[0].segment_id, "ok",
        )

        for seg in batch:
            raw = batch_result.per_segment.get(seg.segment_id)
            if raw is None:
                log.warning("模型未回傳 %s 的結果，標記為 null", seg.segment_id)
                seg.understanding = None
                continue
            understanding = to_understanding(raw, seg, p)
            seg.understanding = understanding
            results.append(understanding)
            prev_summary = understanding.summary[: p.prev_summary_max_chars]
            _save(work, seg, understanding)

    log.info("S4 %s：%d/%d 個 segment 完成。%s",
             work.video_id, len(results), len(segments), ledger.summary())
    return results


def _index_of(work: WorkPaths, segment: Segment) -> int:
    return int(segment.segment_id.rsplit("#", 1)[-1])


def _load_cached(work: WorkPaths, segment: Segment, cfg) -> Understanding | None:
    """讀取既有結果。冪等鍵含 prompt_version 與 model——換了就得重跑（§4.7）。"""
    path = work.understanding(_index_of(work, segment))
    if not path.exists():
        return None
    try:
        cached = Understanding.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 壞掉的快取就當作沒有
        return None
    if cached.model_used != cfg.model or cached.prompt_version != cfg.prompt_version:
        return None
    segment.understanding = cached
    return cached


def _save(work: WorkPaths, segment: Segment, understanding: Understanding) -> None:
    """每 segment 一檔，便於斷點續跑（§3.1、§6.3）。"""
    path = work.understanding(_index_of(work, segment))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(understanding.model_dump_json(indent=2), encoding="utf-8")


SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "tldr": {"type": "string"},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "t_start": {"type": "number"},
                    "t_end": {"type": "number"},
                    "summary": {"type": "string"},
                },
                "required": ["title", "t_start", "t_end", "summary"],
            },
        },
    },
    "required": ["tldr", "chapters"],
}

SYNTHESIS_PROMPT = """你會拿到一支講經影片逐段的理解結果（摘要與術語）。
請據此產出全片層級的統整。

規則：
1. **只能使用下方提供的內容**，不得補充任何外部知識或推測。
2. TL;DR 用三到五句話說明全片講了什麼，寫給沒看過影片的人。
3. 章節依內容主題劃分，時間範圍必須落在提供的 segment 時間內，且
   彼此不重疊、依時間遞增。
4. 不要使用 markdown 裝飾。
"""


def s5_synthesize(cfg: Config, work: WorkPaths, ir: VideoIR) -> VideoIR:
    """S5 全片統整。SDD §4.8。只讀 S4 輸出，**不再讀圖**。成本 1–2 次呼叫。"""
    from ..quota import QuotaExhausted, QuotaLedger
    from .understand import _client

    have = [s for s in ir.segments if s.understanding is not None]
    if not have:
        log.warning("S5 %s：沒有任何理解結果可統整", work.video_id)
        return ir

    # 術語總表可以純本機算出來，不必花額度
    seen: dict[str, int] = {}
    for seg in have:
        for term in seg.understanding.terms:
            seen[term] = seen.get(term, 0) + 1
    ir.term_index = [t for t, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))]

    ledger = QuotaLedger(OutPaths(cfg.out_dir).quota_db, cfg.quota)
    try:
        ledger.check(planned_requests=1, model=cfg.s5.model)
    except QuotaExhausted as exc:
        log.warning("S5 %s：%s；TL;DR 與章節留待下次", work.video_id, exc)
        return ir

    body = "\n\n".join(
        f"## {s.segment_id}（{s.t_start:.0f}s – {s.t_end:.0f}s）\n{s.understanding.summary}"
        for s in have
    )

    from google.genai import types

    try:
        response = _client().models.generate_content(
            model=cfg.s5.model,
            contents=[f"影片標題：{ir.meta.title}\n\n{body}"],
            config=types.GenerateContentConfig(
                system_instruction=SYNTHESIS_PROMPT,
                response_mime_type="application/json",
                response_schema=SYNTHESIS_SCHEMA,
                temperature=0.2,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        # 統整失敗不該讓已花額度的 S4 結果作廢——TL;DR 是加分項，
        # 逐段理解才是產品的主體。
        log.error("S5 %s 失敗，保留逐段結果並繼續：%s", work.video_id, exc)
        ledger.record(cfg.s5.model, 0, 0, None, "error")
        return ir

    import json as _json

    payload = _json.loads(response.text)
    usage = getattr(response, "usage_metadata", None)
    ledger.record(
        cfg.s5.model,
        getattr(usage, "prompt_token_count", 0) or 0,
        getattr(usage, "candidates_token_count", 0) or 0,
        None, "ok",
    )

    ir.tldr = (payload.get("tldr") or "").strip() or None
    ir.chapters = [c for c in payload.get("chapters", []) if c.get("title")]
    log.info("S5 %s：TL;DR + %d 章節 + %d 個術語",
             work.video_id, len(ir.chapters), len(ir.term_index))
    return ir


def s6_render(cfg: Config, ir: VideoIR, work: WorkPaths, out: OutPaths) -> list[Chunk]:
    """S6 渲染。SDD §4.9。

    chunk 切分規則：一個 content_block = 一個 chunk；超過 800 字則按句切分，
    但每個切片都要複製完整 metadata。
    """
    from ..validation.provenance import check_video
    from .render import build_chunks, write_chunks, write_debug_markdown, write_unverified

    out.ensure_dirs()

    # §5.4 溯源檢查（防幻覺閘門）。這一步會就地填回每個 block 的
    # verification 與 similarity，也決定整支影片是否 needs_review。
    verdict = check_video(ir, cfg.provenance)
    write_unverified(verdict, out.unverified)

    chunks, warnings = build_chunks(ir, cfg.s6)
    for warning in warnings:
        log.warning("S6 %s：%s", work.video_id, warning)

    if cfg.s6.write_debug_markdown:
        write_debug_markdown(ir, work, out.debug_md(ir.meta.video_id))

    if ir.needs_review:
        # §5.4：unverified 比例 > 5% → 整支標記 needs_review，**不進 chunks.jsonl**
        from ..validation.thresholds import MAX_UNVERIFIED_RATIO

        log.error(
            "S6 %s：溯源未通過比例 %.1f%%（上限 %.0f%%），整支標記 needs_review，"
            "不寫入 chunks.jsonl。請查看 %s",
            work.video_id, ir.unverified_ratio * 100,
            MAX_UNVERIFIED_RATIO * 100, out.debug_md(ir.meta.video_id),
        )
        return []

    written = write_chunks(chunks, out.chunks)
    log.info("S6 %s：寫出 %d 個 chunk（溯源通過率 %.1f%%）",
             work.video_id, written, verdict.pass_rate * 100)
    return chunks
