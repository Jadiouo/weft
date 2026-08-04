"""階段編排。SDD §6.3 續跑語意 + §6.4 producer/consumer 分離。

編排層負責的事，與各階段本身無關：
  - 依 state.json 判斷哪些階段可跳過（參數 hash 未變且已完成）
  - 參數 hash 變更時作廢下游（`VideoState.invalidate_downstream`）
  - 中途失敗時保留已完成的階段，讓下次執行接得上
"""

from __future__ import annotations

import json
import logging

from .config import Config
from .ir import CandidateSet, Lexicon, Slide, Transcript, VideoMeta
from .paths import OutPaths, WorkPaths
from .state import Stage, StageStatus, VideoState

log = logging.getLogger(__name__)


def stage_params(cfg: Config, stage: Stage) -> str:
    """該階段的參數 hash。上游階段的 hash 不納入——SDD §6.3 的規則是
    「參數 hash 變更 → 該階段及其下游需重跑」，作廢由 invalidate_downstream
    負責，不靠把上游 hash 混進來間接達成。"""
    mapping = {
        Stage.S0_FETCH: cfg.s0,
        Stage.S1A_TRANSCRIPT: cfg.s1a,
        Stage.S1B_SLIDES: cfg.s1b,
        Stage.S2_OCR: cfg.s2,
        Stage.S2B_LEXICON: cfg.s2b,
        Stage.S2C_CORRECT: cfg.s2c,
        Stage.S3_ALIGN: cfg.s3,
        Stage.S4_UNDERSTAND: cfg.s4,
        Stage.S5_SYNTHESIZE: cfg.s5,
        Stage.S6_RENDER: cfg.s6,
    }
    return mapping[stage].params_hash()


def sync_state(cfg: Config, state: VideoState) -> set[Stage]:
    """比對設定與 state.json，作廢所有參數已變更階段的下游。回傳被作廢的階段。"""
    invalidated: set[Stage] = set()
    for stage in Stage:
        rec = state.stages.get(stage)
        if rec is None or rec.params_hash is None:
            continue
        if rec.params_hash != stage_params(cfg, stage):
            log.info("%s 的參數已變更，作廢其下游", stage.value)
            invalidated |= state.invalidate_downstream(stage)
            state.stages[stage].params_hash = None
    return invalidated


def resolve_targets(target: str) -> list[tuple[str, str | None, int | None]]:
    """把 playlist URL / video URL / video_id 解析成
    `[(video_id, series_id, episode_index)]`。"""
    from .stages.fetch import expand_playlist, parse_target

    video_id, playlist_id = parse_target(target)
    if playlist_id:
        items = expand_playlist(playlist_id)
        log.info("playlist %s 共 %d 支影片", playlist_id, len(items))
        return [(i.video_id, i.series_id, i.episode_index) for i in items]
    if video_id:
        return [(video_id, None, None)]
    raise ValueError(f"無法從 {target!r} 解析出 video_id 或 playlist_id")


def _load_skiplist(out: OutPaths) -> dict:
    if out.skip_list.exists():
        return json.loads(out.skip_list.read_text(encoding="utf-8"))
    return {}


def _record_skip(out: OutPaths, video_id: str, reason: str) -> None:
    """SDD §4.1 失敗行為：影片不可用 → 記入 skip list，繼續下一支。"""
    out.ensure_dirs()
    skips = _load_skiplist(out)
    skips[video_id] = reason
    out.skip_list.write_text(json.dumps(skips, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_one(
    video_id: str,
    cfg: Config,
    series_id: str | None = None,
    episode_index: int | None = None,
    force: bool = False,
) -> VideoState:
    """跑單支影片的 S0–S3。每個階段各自檢查是否可跳過（§6.3）。"""
    from .stages import local

    work = WorkPaths(cfg.work_dir, video_id)
    work.ensure_dirs()
    state = VideoState(video_id=video_id) if force else VideoState.load_or_new(work.state, video_id)
    sync_state(cfg, state)

    def satisfied(stage: Stage) -> bool:
        return state.is_satisfied(stage, stage_params(cfg, stage))

    # ---- S0 取得 ----
    if satisfied(Stage.S0_FETCH) and work.meta.exists():
        meta = VideoMeta.model_validate_json(work.meta.read_text(encoding="utf-8"))
    else:
        meta = local.s0_fetch(video_id, cfg, work)
        if series_id or episode_index:
            meta = meta.model_copy(update={"series_id": series_id, "episode_index": episode_index})
            work.meta.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        state.mark_done(Stage.S0_FETCH, stage_params(cfg, Stage.S0_FETCH))
        state.save(work.state)

    # ---- S1b 投影片（先於 S1a：詞庫要餵給 Whisper 的 initial_prompt）----
    if satisfied(Stage.S1B_SLIDES) and work.candidates.exists():
        candidates = CandidateSet.model_validate_json(work.candidates.read_text(encoding="utf-8"))
        slides = _slides_from(work, candidates)
    else:
        candidates, slides = local.s1b_slides(cfg, work)
        state.mark_done(Stage.S1B_SLIDES, stage_params(cfg, Stage.S1B_SLIDES))
        state.save(work.state)

    # ---- S2 OCR ----
    if satisfied(Stage.S2_OCR) and work.ocr.exists():
        _apply_ocr(work, slides)
    else:
        slides = local.s2_ocr(cfg, work, slides)
        state.mark_done(Stage.S2_OCR, stage_params(cfg, Stage.S2_OCR))
        state.save(work.state)

    # ---- S2b 詞庫（系列級）----
    if satisfied(Stage.S2B_LEXICON) and work.lexicon.exists():
        lexicon = Lexicon.model_validate_json(work.lexicon.read_text(encoding="utf-8"))
    else:
        lexicon = local.s2b_lexicon(cfg, work, slides, meta.series_id or series_id)
        state.mark_done(Stage.S2B_LEXICON, stage_params(cfg, Stage.S2B_LEXICON))
        state.save(work.state)

    # ---- S1a 逐字稿 ----
    if satisfied(Stage.S1A_TRANSCRIPT) and work.transcript.exists():
        transcript = Transcript.model_validate_json(work.transcript.read_text(encoding="utf-8"))
    else:
        transcript = local.s1a_transcript(cfg, work, lexicon)
        state.mark_done(Stage.S1A_TRANSCRIPT, stage_params(cfg, Stage.S1A_TRANSCRIPT))
        state.save(work.state)

    # ---- S2c 術語校正 ----
    if not satisfied(Stage.S2C_CORRECT):
        transcript = local.s2c_correct(cfg, work, transcript, lexicon)
        note = "詞庫為空，已跳過（§4.5）" if not lexicon.entries else None
        state.mark_done(Stage.S2C_CORRECT, stage_params(cfg, Stage.S2C_CORRECT), note=note)
        if not lexicon.entries:
            state.stages[Stage.S2C_CORRECT].status = StageStatus.SKIPPED
        state.save(work.state)

    # ---- S3 對齊 ----
    if not satisfied(Stage.S3_ALIGN):
        local.s3_align(cfg, work, transcript, candidates)
        state.mark_done(Stage.S3_ALIGN, stage_params(cfg, Stage.S3_ALIGN))
        state.save(work.state)

    return state


def _slides_from(work: WorkPaths, candidates: CandidateSet) -> list[Slide]:
    """從既有的 02_candidates.json 重建 Slide 物件（續跑用）。"""
    slides = []
    for c in candidates.candidates:
        slide_id = f"slide_{c.index + 1:03d}"
        slides.append(
            Slide(
                slide_id=slide_id,
                image_path=str(work.slide_image(c.index + 1).relative_to(work.dir)),
                t_first_seen=c.t_start,
                t_last_seen=c.t_end,
                is_progressive_final=bool(c.build_frames),
                build_frames=c.build_frames,
            )
        )
    _apply_ocr(work, slides)
    return slides


def _apply_ocr(work: WorkPaths, slides: list[Slide]) -> None:
    if not work.ocr.exists():
        return
    rows = {r["slide_id"]: r for r in json.loads(work.ocr.read_text(encoding="utf-8"))}
    for slide in slides:
        row = rows.get(slide.slide_id)
        if row:
            slide.ocr_text = row.get("ocr_text")
            slide.ocr_confidence = row.get("ocr_confidence")


def run_prepare(target: str, cfg: Config, force: bool = False) -> int:
    """S0–S3。不受額度限制，可一次跑完整個系列。SDD §6.4。"""
    from .stages.fetch import VideoUnavailable

    out = OutPaths(cfg.out_dir)
    targets = resolve_targets(target)
    skips = _load_skiplist(out)

    failed = 0
    for video_id, series_id, episode_index in targets:
        if video_id in skips:
            log.info("%s 在 skip list 中（%s），跳過", video_id, skips[video_id])
            continue
        try:
            prepare_one(video_id, cfg, series_id, episode_index, force=force)
        except VideoUnavailable as exc:
            # §4.1：影片不可用 → 記入 skip list，繼續下一支
            log.warning("%s 不可用，記入 skip list：%s", video_id, exc)
            _record_skip(out, video_id, str(exc))
        except Exception as exc:  # noqa: BLE001
            # §4.2／§4.3：單支失敗標記 failed 並繼續——批次跑數十支時，
            # 中途失敗是常態而非例外（§2.1 原則四）
            failed += 1
            log.exception("%s 處理失敗，繼續下一支：%s", video_id, exc)
            work = WorkPaths(cfg.work_dir, video_id)
            state = VideoState.load_or_new(work.state, video_id)
            state.mark_failed(Stage.S0_FETCH, str(exc))
            state.save(work.state)

    log.info("prepare 完成：%d 支目標，%d 支失敗", len(targets), failed)
    return 1 if failed == len(targets) and targets else 0


def run_understand(cfg: Config, video_id: str | None = None, max_requests: int | None = None) -> int:
    """S4–S6。消化 prepare 囤下的 buffer，額度耗盡即停。SDD §6.4。"""
    from .stages import cloud

    work = WorkPaths(cfg.work_dir, video_id or "")
    cloud.s4_understand(cfg, work, [], None)  # type: ignore[arg-type]
    return 0
