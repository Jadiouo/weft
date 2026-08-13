"""S4–S6：理解、統整、渲染。SDD §4.7–§4.9，屬 Phase 2–3（§7.3–§7.4）。

S4 是**唯一花額度的階段**。額度來源固定為 Gemini API key（AI Studio free
tier）；§5.5 #12 明文禁止把 AI Pro/Ultra 的訂閱額度接入程式化呼叫。
"""

from __future__ import annotations

import logging

from ..config import Config
from ..ir import Chunk, Segment, SegmentMode, Slide, Understanding, VideoIR
from ..paths import OutPaths, WorkPaths
from . import pending

log = logging.getLogger(__name__)


def _batches(segments: list[Segment], size: int) -> list[list[Segment]]:
    """把相鄰 segment 分組。SDD §4.7：可將 2–3 個合併為一次呼叫節省額度。"""
    size = max(1, size)
    return [segments[i : i + size] for i in range(0, len(segments), size)]


#: 投影片段落中「沒有任何 content_block」的比例上限。
#: **不是校準值，是保護性上限。** 正常執行實測是 0%（每個投影片段落都
#: 至少產出一個 block）；壞掉那次是 70%（zIglvjoU9vo 33 段中 23 段空的）。
#: 中間沒有樣本，所以 0.3 取的是「明顯不正常」而非某個分界。
_MAX_BARREN_RATIO = 0.3
#: 少於這個數量就不算比例。**1 個樣本算不出比例**——單元測試的假模型
#: 只有 1 個投影片段落，比例必然是 0% 或 100%，兩者都不代表什麼。
#: 代價：短到只有 4 個投影片段落的影片不受這道保護。可接受，
#: 因為那種影片人工掃一眼就看得完。
_MIN_SLIDE_SEGMENTS_FOR_BARREN_CHECK = 5


def s4_understand(
    cfg: Config,
    work: WorkPaths,
    segments: list[Segment],
    slides: list[Slide],
    transcript=None,
) -> list[Understanding]:
    """S4 聯合理解。SDD §4.7。

    **v0.3 起 S4 多了兩項職責**（原本由 S2/S2b/S2c 的本地 OCR 鏈負責）：
      1. 判斷候選幀是不是投影片（`is_slide`）——不是就把該段降級為
         speaker_only 並清掉 slide_ref
      2. 對照投影片修正逐字稿的同音錯字，回傳可稽核的 corrections

    冪等鍵：segment_id + prompt_version + model
    失敗行為：單一 segment 失敗 → 重試 2 次（指數退避）→ understanding=null，繼續
    """
    from ..quota import QuotaExhausted, QuotaLedger
    from .providers import costs_quota
    from .understand import (
        ApiKeyMissing,
        PermanentApiError,
        QuotaHit,
        apply_corrections,
        call_model,
        to_understanding,
        with_retries,
    )

    p = cfg.s4
    ledger = QuotaLedger(OutPaths(cfg.out_dir).quota_db, cfg.quota)
    #: 本地模型不佔雲端額度（§2.3、§6.5）。**不能無條件記帳**——
    #: 實測全本地配置跑到一半被自己的帳本判定「額度用盡」而停下。
    metered = costs_quota(p.model)
    work.understanding_dir.mkdir(parents=True, exist_ok=True)

    by_slide_id = {s.slide_id: s for s in slides}
    image_paths = {s.slide_id: work.dir / s.image_path for s in slides}
    # v0.4：S4a 已讀出投影片文字，這裡以**文字**掛進 prompt（§4.7c）
    slide_context = {
        s.slide_id: {"slide_text": s.slide_text, "description": s.layout_description}
        for s in slides if s.slide_text or s.layout_description
    }

    results: list[Understanding] = []
    prev_summary: str | None = None

    for batch in _batches(segments, p.batch_segments):
        # 續跑：整批都已有結果就直接讀檔，不重複花額度（§6.3）
        cached = [_load_cached(work, seg, p) for seg in batch]
        if all(c is not None for c in cached):
            # 快取命中也要**重建衍生狀態**（D22）。原本這裡直接 continue，
            # 於是續跑時 `mode` 降級與 `text_corrected` 都沿用磁碟上的舊值——
            # 前一次若寫壞了就永遠壞著。衍生狀態必須是快取的純函數。
            for seg, understanding in zip(batch, cached, strict=True):
                if understanding is None:
                    continue
                _apply_to_segment(seg, understanding, by_slide_id, transcript)
            results += [c for c in cached if c is not None]
            prev_summary = results[-1].summary if results else prev_summary
            continue

        # §6.1 主動節流：呼叫**前**估算，不靠撞 429（§5.5 #13）
        if metered:
            try:
                ledger.check(planned_requests=1, model=p.model)
            except QuotaExhausted as exc:
                log.warning("%s；停止本日處理，進度已保存（§6.1）", exc)
                break

        # 每一次**實際 API 呼叫**都要記帳，不是每個批次記一次——重試也
        # 消耗配額。v0.3 首跑時記「批次」，帳本顯示 17 次而實際打了 45+ 次。
        def _record_attempt(ok: bool, exc, _seg=batch[0].segment_id):
            if metered and not ok:
                ledger.record(p.model, 0, 0, _seg, "error")

        try:
            batch_result = with_retries(
                lambda b=batch: call_model(b, image_paths, prev_summary, p,
                                           slide_context),
                p.max_retries,
                p.retry_backoff_sec,
                on_attempt=_record_attempt,
            )
        except ApiKeyMissing:
            raise
        except QuotaHit as exc:
            # §6.1：撞到 429 代表主動節流的估算錯了。把 API 回報的真實配額
            # 記進帳本，下次才不會重蹈覆轍（SDD §9 的緩解措施）。
            if exc.limit:
                ledger.record_observed_limit(p.model, exc.limit)
            log.warning("撞到配額上限，停止本日處理（進度已保存）：%s",
                        str(exc)[:200])
            break
        except PermanentApiError as exc:
            # 模型不存在、請求格式錯——**整批都會失敗，繼續下去只是白燒配額**。
            # v0.3 首跑時模型名已停用，15 個批次各撞 3 次才停下來。
            log.error("永久性錯誤，立即中止（不再嘗試後續批次）：%s", exc)
            for seg in batch:
                seg.understanding = None
            break
        except Exception as exc:  # noqa: BLE001
            # §4.7 失敗行為：重試後仍失敗 → understanding=null，繼續下一批
            log.error("批次 %s 失敗，標記為 null 並繼續：%s",
                      [s.segment_id for s in batch], exc)
            for seg in batch:
                seg.understanding = None
            continue

        if metered:
            ledger.record(
                p.model, batch_result.input_tokens, batch_result.output_tokens,
                batch[0].segment_id, "ok",
            )

        missing = [s for s in batch if s.segment_id not in batch_result.per_segment]
        if missing and len(batch) > 1:
            # **批次沒回齊就逐段補跑**，不要直接標 null。
            # 實測本地模型在 batch=3 時偶爾只回 2 段——那不是內容問題，
            # 是它沒把陣列生完。逐段重問一次通常就有了。
            log.info("批次少回 %d 段，逐段補跑：%s",
                     len(missing), [s.segment_id for s in missing])
            for seg in missing:
                try:
                    single = with_retries(
                        lambda one=seg: call_model([one], image_paths, prev_summary, p,
                                                   slide_context),
                        p.max_retries, p.retry_backoff_sec, on_attempt=_record_attempt,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s 逐段補跑仍失敗：%s", seg.segment_id, str(exc)[:120])
                    continue
                if metered:
                    ledger.record(p.model, single.input_tokens, single.output_tokens,
                                  seg.segment_id, "ok")
                batch_result.per_segment.update(single.per_segment)

        for seg in batch:
            raw = batch_result.per_segment.get(seg.segment_id)
            if raw is None:
                log.warning("模型未回傳 %s 的結果，標記為 null", seg.segment_id)
                seg.understanding = None
                continue
            slide_obj = by_slide_id.get(seg.slide_ref or seg.candidate_ref or "")
            understanding = to_understanding(raw, seg, p, slide_obj=slide_obj)
            _apply_to_segment(seg, understanding, by_slide_id, transcript)

            results.append(understanding)
            prev_summary = understanding.summary[: p.prev_summary_max_chars]
            _save(work, seg, understanding, p)

    confirmed = sum(1 for u in results if u.is_slide)
    corrected = sum(len(u.corrections) for u in results)

    # **投影片段落卻一個 content_block 都沒有，是壞掉不是「這頁沒東西可講」。**
    #
    # 實測（2026-08-08）：改壞 S4c 的 prompt 之後，四支影片的 block 從 167 個
    # 掉到 21 個，而**每一項機械檢查都是綠的**——溯源通過率甚至更好看，
    # 因為分母小了。管線照常產出一份少了 87% 內容的知識庫。
    # 與 D22 的 rehydrate 靜靜回傳空的、S4a 的全軍覆沒同一類：
    # 輸出「變少」不會自己報錯，必須主動檢查。
    slide_segments = [u for u in results if u.is_slide]
    barren = [u for u in slide_segments if not u.content_blocks]
    if (len(slide_segments) >= _MIN_SLIDE_SEGMENTS_FOR_BARREN_CHECK
            and len(barren) / len(slide_segments) > _MAX_BARREN_RATIO):
        raise RuntimeError(
            f"S4c {work.video_id}：{len(slide_segments)} 個投影片段落中有 "
            f"{len(barren)} 個（{len(barren) / len(slide_segments):.0%}）"
            f"沒有產出任何 content_block，超過上限 {_MAX_BARREN_RATIO:.0%}。"
            f"這通常是 prompt 或 schema 壞了，不是素材沒東西可講。"
            f"已中止，不產出殘缺的知識庫。"
        )
    log.info("S4c %s：%d/%d 個 segment 完成（%d 個判定為投影片，%d 筆術語校正）。%s",
             work.video_id, len(results), len(segments), confirmed, corrected,
             ledger.summary(p.model) if metered else f"本地模型 {p.model}，不佔額度")
    return results


def _apply_to_segment(seg, understanding, by_slide_id, transcript) -> None:
    """把一份 Understanding 的效果套到 segment / slide / transcript 上。

    **快取命中與新呼叫都要走這裡**（D22）。這些都是 Understanding 的
    衍生狀態，必須可以從快取重建；只在「新呼叫」那條路上做的話，
    續跑會沿用磁碟上的舊值，前一次寫壞了就永遠壞著。
    """
    from .understand import apply_corrections

    seg.understanding = understanding

    # VLM 判定不是投影片 → 降級。slide_ref 清掉（沒有投影片可指向），
    # candidate_ref 保留，讓 debug markdown 還能顯示被拒絕的那張圖。
    # v0.4：`is_slide` 由 **S4a** 判定（§4.7a），這裡只是把結果反映到 segment。
    # 不再回頭寫 `slide.slide_text`——那是 S4a 的產出，S4c 不該覆寫它。
    if not understanding.is_slide:
        seg.mode = SegmentMode.SPEAKER_ONLY
        seg.slide_ref = None

    _canonicalize_refs(seg, understanding)

    if understanding.corrections and transcript is not None:
        apply_corrections(transcript, seg, understanding.corrections)


def _canonicalize_refs(seg, understanding) -> None:
    """把 `provenance.ref` 換成管線的權威值。

    **這個欄位本來就不該由模型提供。** 一段只會拿到一張投影片的文字
    （`slide_context` 只掛那一張），所以 `slide_ocr` 的合法來源只有一個；
    `transcript` 的區間就是這一段自己的起訖。模型寫的是它抄下來的字串。

    實測（C1，四支影片 28 筆未通過）：**3 筆是模型把 `slide_015` 寫成
    `015`**，查不到投影片 → 來源字串為空 → source_ratio 0% → 未通過，
    而且失敗原因顯示「來源長度僅 0%」，看起來像內容太長，其實是查無此人。
    內容本身是忠實的。

    這不是模糊比對（§5.3 不變量 10 禁止的那種）——是**改用權威值取代猜測**。
    模型指到別張投影片時記一筆 warning：那是它搞混了，值得看。
    """
    from ..ir import ProvenanceKind

    if understanding is None:
        return
    slide_id = seg.slide_ref or seg.candidate_ref
    for i, block in enumerate(understanding.content_blocks):
        if block.provenance.kind is ProvenanceKind.SLIDE_OCR:
            if not slide_id or block.provenance.ref == slide_id:
                continue
            log.warning("%s 的 block#%d 指向 %r，改為本段的 %s",
                        seg.segment_id, i, block.provenance.ref, slide_id)
            block.provenance.ref = slide_id
        else:
            block.provenance.ref = f"{seg.t_start:.1f}s-{seg.t_end:.1f}s"


def _index_of(work: WorkPaths, segment: Segment) -> int:
    return int(segment.segment_id.rsplit("#", 1)[-1])


def segment_fingerprint(segment: Segment) -> str:
    """這個 segment 的**輸入**指紋：逐字稿原文 + 指到的投影片 + 時間範圍。

    §4.7 的冪等鍵原本只有 `segment_id + prompt_version + model`，
    而 `segment_id` 是位置編號。**換一種分段方式，同一個編號涵蓋的內容
    就完全不同**，舊快取卻照樣命中。實測 v0.5 換成逐字稿主幹之後，
    cxrqHABhWOU 的 `#010` 從 72–98 秒變成 564–593 秒，理解結果卻沿用舊的
    ——溯源通過率 0.929 → 0.071，而每一項機械檢查都是綠的。

    用 `transcript_raw` 而不是 corrected：後者會被 S4b 改，
    而改了不代表要重新理解。時間範圍也納入，因為同樣的文字落在不同
    時段時 `slide_ref` 可能不同。
    """
    import hashlib

    h = hashlib.sha256()
    h.update(f"{segment.t_start:.3f}|{segment.t_end:.3f}|".encode())
    h.update((segment.slide_ref or segment.candidate_ref or "").encode())
    h.update(b"|")
    h.update(segment.transcript_raw.encode())
    return h.hexdigest()[:16]


def sampling_fingerprint(cfg) -> str:
    """所有影響取樣的參數的指紋。**加新參數只要改這裡。**

    這個 repo 已經六次因為「冪等鍵沒涵蓋某個決定結果的東西」而量出
    假結論（D20／D22／D30／D32／`DEPTH_ALPHA`／`temperature`）。
    逐個欄位比對的寫法要求每次加參數都記得改比對邏輯——
    **那個「記得」遲早會失效**，而失效的症狀是「改了沒效果」，
    最難察覺。

    `tests/test_unit_sampling_fingerprint.py` 釘住「每個取樣參數都要
    讓指紋改變」，用的是反射而不是硬編清單——**新加的欄位自動被涵蓋**。
    """
    import hashlib

    parts = "|".join(
        f"{name}={getattr(cfg, name, None)!r}" for name in SAMPLING_PARAMS)
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:16]


#: 影響取樣結果的設定欄位。**新增取樣參數時要加進來**，
#: 而 `test_unit_sampling_fingerprint.py` 會檢查沒有漏掉。
SAMPLING_PARAMS: tuple[str, ...] = ("temperature", "seed", "top_k")


def _load_cached(work: WorkPaths, segment: Segment, cfg) -> Understanding | None:
    """讀取既有結果。冪等鍵含 prompt_version、model **與輸入指紋**（§4.7）。"""
    path = work.understanding(_index_of(work, segment))
    if not path.exists():
        return None
    try:
        cached = Understanding.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 壞掉的快取就當作沒有
        return None
    if cached.model_used != cfg.model or cached.prompt_version != cfg.prompt_version:
        return None
    want = sampling_fingerprint(cfg)
    if cached.sampling_fingerprint != want:
        # 舊快取沒有這個欄位（None）時也走這裡——**保守地重跑**，
        # 理由與下面的輸入指紋相同：不能相信一個無法驗證的假設。
        log.info("%s 的取樣參數變了（%s → %s，溫度 %s → %s），重跑",
                 segment.segment_id, cached.sampling_fingerprint, want,
                 cached.temperature, cfg.temperature)
        return None
    fingerprint = segment_fingerprint(segment)
    if cached.input_fingerprint != fingerprint:
        # 舊快取沒有這個欄位（None）時也走這裡——**保守地重跑**。
        # 讓沒有指紋的快取命中，等於相信一個無法驗證的假設。
        log.info("%s 的輸入變了（或快取來自舊版），重跑", segment.segment_id)
        return None
    segment.understanding = cached
    return cached


def _save(work: WorkPaths, segment: Segment, understanding: Understanding,
          cfg=None) -> None:
    """每 segment 一檔，便於斷點續跑（§3.1、§6.3）。"""
    understanding.input_fingerprint = segment_fingerprint(segment)
    if cfg is not None:
        understanding.temperature = cfg.temperature
        understanding.sampling_fingerprint = sampling_fingerprint(cfg)
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
    """S5 全片統整。SDD §4.8。只讀 S4c 輸出，**不再讀圖**。成本 1–2 次呼叫。"""
    from ..quota import QuotaExhausted, QuotaLedger
    from .providers import Part, costs_quota, generate

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

    metered = costs_quota(cfg.s5.model)
    ledger = QuotaLedger(OutPaths(cfg.out_dir).quota_db, cfg.quota)
    if metered:
        try:
            ledger.check(planned_requests=1, model=cfg.s5.model)
        except QuotaExhausted as exc:
            log.warning("S5 %s：%s；TL;DR 與章節留待下次", work.video_id, exc)
            return ir

    body = "\n\n".join(
        f"## {s.segment_id}（{s.t_start:.0f}s – {s.t_end:.0f}s）\n{s.understanding.summary}"
        for s in have
    )

    try:
        result = generate(
            cfg.s5.model, SYNTHESIS_PROMPT,
            [Part(text=f"影片標題：{ir.meta.title}\n\n{body}")],
            SYNTHESIS_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001
        # S5 失敗不該拖垮整支——逐段結果已經在磁碟上了
        if metered:
            ledger.record(cfg.s5.model, 0, 0, None, "error")
        log.error("S5 %s 失敗，保留逐段結果並繼續：%s", work.video_id, str(exc)[:200])
        return ir

    if metered:
        ledger.record(cfg.s5.model, result.input_tokens, result.output_tokens, None, "ok")

    payload = result.payload
    ir.tldr = (payload.get("tldr") or "").strip()
    ir.chapters = [
        {"title": (c.get("title") or "").strip(), "t_start": float(c.get("t_start") or 0.0)}
        for c in payload.get("chapters", [])
        if (c.get("title") or "").strip()
    ]
    log.info("S5 %s：TL;DR %d 字、章節 %d 個、術語 %d 個",
             work.video_id, len(ir.tldr), len(ir.chapters), len(ir.term_index))
    return ir


def s6_render(cfg: Config, ir: VideoIR, work: WorkPaths, out: OutPaths) -> list[Chunk]:
    """S6 渲染。SDD §4.9。

    chunk 切分規則：一個 content_block = 一個 chunk；超過 800 字則按句切分，
    但每個切片都要複製完整 metadata。
    """
    from ..validation.provenance import check_video
    from ..validation.thresholds import (
        MAX_UNVERIFIED_RATIO,
        MIN_CHARS_PER_1K_SOURCE,
        PROVENANCE_PER_VIDEO_GATE,
    )
    from .render import (
        build_chunks,
        drop_video_from_chunks,
        overall_provenance_rate,
        write_chunks,
        write_debug_markdown,
        write_provenance_record,
        write_unverified,
    )

    out.ensure_dirs()

    # §5.4 溯源檢查（防幻覺閘門）。這一步會就地填回每個 block 的
    # verification 與 similarity，也決定整支影片是否 needs_review。
    verdict = check_video(ir, cfg.provenance)
    write_unverified(verdict, out.unverified)
    # **逐支記錄，不設門檻**（票 03）。閘門是下面那個 per-video 的判斷；
    # 這裡只是把數字與成因留下來，讓「整體趨勢」有東西可看。
    # 產出量要拿逐字稿當分母。S6 的簽名沒有 transcript，從 work 讀；
    # **讀不到就只記速率、不設下限**——新增的觀測值不該讓整條路徑壞掉。
    source_transcript = None
    if work.transcript.exists():
        from ..ir import Transcript

        source_transcript = Transcript.model_validate_json(
            work.transcript.read_text(encoding="utf-8"))
    record = write_provenance_record(
        verdict, out.provenance_log, ir, source_transcript)

    # **產出量的災難下限。** 通過率上升有可能只是因為寫得比較少——
    # 2026-08-09 實測 `depth_alpha` 一改，`2FjApOVIbUs` 的總字數掉 47%
    # 而通過率從 0.979 升到 1.000。D31 的 `_MAX_BARREN_RATIO` 擋得住
    # 「整段空白」，擋不住「每段都少寫一半」。
    #
    # 這**不是品質門檻**（各素材合理值差很多，實測 70–176 字/分），
    # 是「S4c 幾乎沒在寫東西」的偵測器。
    ratio = record.get("chars_per_1k_source")
    if ratio is not None and ratio < MIN_CHARS_PER_1K_SOURCE:
        raise RuntimeError(
            f"S4c 產出量過低：每千字逐字稿只產出 {ratio} 字"
            f"（下限 {MIN_CHARS_PER_1K_SOURCE}）。"
            f"{record['segments']} 段產出 {record['blocks']} 個 block。"
            f"**先確認 S4c 是不是幾乎沒回東西**，不要直接調低這個下限。"
        )

    chunks, warnings = build_chunks(ir, cfg.s6)
    for warning in warnings:
        log.warning("S6 %s：%s", work.video_id, warning)

    if cfg.s6.write_debug_markdown:
        write_debug_markdown(ir, work, out.debug_md(ir.meta.video_id))

    if ir.needs_review:
        # **D34：這裡不再丟掉整支影片。** 未通過的 block 已由 `build_chunks`
        # 逐塊排除（§5.4 第一層）；per-video 這一層原本是「整支丟掉」，
        # 而實測 8 支素材時它丟掉的已驗證內容（218 block）比留下的（135）還多。
        #
        # 改為大聲記錄 + 把通過率寫進每個 chunk 的 metadata，
        # 讓下游自己決定信任門檻。理由與代價見 docs/decisions.md D34。
        log.error(
            "S6 %s：溯源未通過比例 %.1f%%（標記門檻 %.0f%%），整支標記 "
            "needs_review。**已驗證的 block 仍會輸出**，chunk 的 "
            "video_pass_rate=%.3f。請查看 %s",
            work.video_id, ir.unverified_ratio * 100,
            MAX_UNVERIFIED_RATIO * 100, 1.0 - ir.unverified_ratio,
            out.debug_md(ir.meta.video_id),
        )

    written = write_chunks(chunks, out.chunks, video_id=ir.meta.video_id)
    overall = overall_provenance_rate(out.provenance_log)
    log.info("S6 %s：寫出 %d 個 chunk（本支溯源 %.1f%%，標記門檻 %.0f%%）；"
             "累計 %d 支、整體 %.1f%%（**只是趨勢，不是驗收門檻**）",
             work.video_id, written, verdict.pass_rate * 100,
             PROVENANCE_PER_VIDEO_GATE * 100, overall["videos"],
             overall["provenance_rate_overall"] * 100)
    return chunks
