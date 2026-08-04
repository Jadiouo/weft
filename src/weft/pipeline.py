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
from .ir import CandidateSet, Slide, Transcript, VideoMeta
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

    # ---- S1b 靜止區段 ----
    if satisfied(Stage.S1B_SLIDES) and work.candidates.exists():
        candidates = CandidateSet.model_validate_json(work.candidates.read_text(encoding="utf-8"))
        slides = _slides_from(work, candidates)
    else:
        candidates, slides = local.s1b_slides(cfg, work)
        state.mark_done(Stage.S1B_SLIDES, stage_params(cfg, Stage.S1B_SLIDES))
        state.save(work.state)

    # ---- S1a 逐字稿 ----
    if satisfied(Stage.S1A_TRANSCRIPT) and work.transcript.exists():
        transcript = Transcript.model_validate_json(work.transcript.read_text(encoding="utf-8"))
    else:
        transcript = local.s1a_transcript(cfg, work)
        state.mark_done(Stage.S1A_TRANSCRIPT, stage_params(cfg, Stage.S1A_TRANSCRIPT))
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
    return slides


def run_survey(target: str, cfg: Config, sample: int = 3) -> int:
    """S-1 素材勘查。SDD §4.0。**新系列開跑前必跑。**

    只抽樣數支——目的是回答「這個系列能不能用現有設計處理」，不是逐支驗證。
    不符時回傳非 0 並列出原因，**不自動繼續**。
    """
    from .stages.fetch import VideoUnavailable
    from .stages.survey import survey, write_profile

    out = OutPaths(cfg.out_dir)
    targets = resolve_targets(target)
    series_id = next((s for _, s, _ in targets if s), None)
    picked = [v for v, _, _ in targets[:sample]]

    # 勘查需要影片本身，所以缺的先取得（只取抽樣的那幾支）
    for video_id in picked:
        work = WorkPaths(cfg.work_dir, video_id)
        if work.video.exists():
            continue
        work.ensure_dirs()
        try:
            from .stages import local

            local.s0_fetch(video_id, cfg, work)
        except VideoUnavailable as exc:
            log.warning("%s 不可用，勘查略過：%s", video_id, exc)

    profile = survey(picked, cfg, series_id)
    path = write_profile(profile, cfg.out_dir)
    log.info("S-1 profile 寫入 %s", path)

    if profile.ok:
        log.info("S-1：%d 支抽樣影片與 §1.3 的 profile 相符，可以繼續 prepare。",
                 len(profile.videos))
        return 0

    log.error("S-1：**素材與現有設計的假設不符，已中止**（SDD §4.0）")
    for problem in profile.mismatches:
        log.error("  - %s", problem)
    log.error("詳見 %s。若確定要繼續，請先修改 SDD §1.3／§4.3 的設計。", path)
    return 3


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


def _ready_for_understanding(cfg: Config) -> list[str]:
    """掃描 work/，找出已跑完 prepare、可進 S4 的影片。SDD §6.3 重建待辦佇列。"""
    if not cfg.work_dir.exists():
        return []
    ready: list[str] = []
    for video_dir in sorted(p for p in cfg.work_dir.iterdir() if p.is_dir()):
        work = WorkPaths(cfg.work_dir, video_dir.name)
        if not (work.state.exists() and work.segments.exists()):
            continue
        state = VideoState.load(work.state)
        if state.is_satisfied(Stage.S3_ALIGN, stage_params(cfg, Stage.S3_ALIGN)):
            ready.append(video_dir.name)
    return ready


def understand_one(video_id: str, cfg: Config) -> bool:
    """單支影片的 S4–S6。回傳是否完整跑完（False = 額度用盡而中止）。"""
    from .ir import Segment, VideoIR
    from .quota import QuotaExhausted, QuotaLedger
    from .stages import cloud

    work = WorkPaths(cfg.work_dir, video_id)
    out = OutPaths(cfg.out_dir)
    state = VideoState.load_or_new(work.state, video_id)
    sync_state(cfg, state)

    meta = VideoMeta.model_validate_json(work.meta.read_text(encoding="utf-8"))
    segments = [
        Segment.model_validate(row)
        for row in json.loads(work.segments.read_text(encoding="utf-8"))
    ]
    candidates = CandidateSet.model_validate_json(work.candidates.read_text(encoding="utf-8"))
    slides = _slides_from(work, candidates)
    transcript = Transcript.model_validate_json(work.transcript.read_text(encoding="utf-8"))

    # ---- S4 理解（含 is_slide 判定與術語校正）----
    cloud.s4_understand(cfg, work, segments, slides, transcript)
    done = [s for s in segments if s.understanding is not None]
    state.understood_segments = [s.segment_id for s in done]

    if len(done) < len(segments):
        # 額度用盡或部分失敗——保存進度，不推進到 S5/S6。
        # §6.3：重啟時掃描 work/，從中斷處繼續。
        state.save(work.state)
        log.info("%s：%d/%d segment 完成，保留進度待續跑",
                 video_id, len(done), len(segments))
        return False

    state.mark_done(Stage.S4_UNDERSTAND, stage_params(cfg, Stage.S4_UNDERSTAND))
    state.save(work.state)

    # S4 可能改過逐字稿的 text_corrected（術語校正），落地
    work.transcript.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    ir = VideoIR(meta=meta, slides=slides, segments=segments)

    # ---- S5 統整 ----
    ir = cloud.s5_synthesize(cfg, work, ir)
    state.mark_done(Stage.S5_SYNTHESIZE, stage_params(cfg, Stage.S5_SYNTHESIZE))

    # ---- S6 渲染（含 §5.4 溯源閘門）----
    cloud.s6_render(cfg, ir, work, out)
    work.video_ir.write_text(ir.model_dump_json(indent=2), encoding="utf-8")
    state.mark_done(Stage.S6_RENDER, stage_params(cfg, Stage.S6_RENDER))
    state.save(work.state)
    return True


def run_understand(cfg: Config, video_id: str | None = None, max_requests: int | None = None) -> int:
    """S4–S6。消化 prepare 囤下的 buffer，額度耗盡即停。SDD §6.4。"""
    from .quota import QuotaExhausted, QuotaLedger
    from .stages.understand import ApiKeyMissing

    out = OutPaths(cfg.out_dir)
    ledger = QuotaLedger(out.quota_db, cfg.quota)
    targets = [video_id] if video_id else _ready_for_understanding(cfg)
    if not targets:
        log.warning("沒有已就緒的影片。請先跑 `weft prepare <playlist|video>`。")
        return 1

    log.info("understand：%d 支待處理。%s", len(targets), ledger.summary())
    completed = 0
    for vid in targets:
        if max_requests is not None and ledger.usage_today().requests >= max_requests:
            log.info("已達本次執行的請求上限 %d，停止", max_requests)
            break
        try:
            if understand_one(vid, cfg):
                completed += 1
            else:
                # 額度用盡——後面的影片同樣跑不動，不必逐支重試
                log.info("額度用盡，停止本日處理。下次重置：%s", ledger.next_reset().isoformat())
                break
        except ApiKeyMissing as exc:
            log.error("%s", exc)
            return 2
        except QuotaExhausted as exc:
            log.info("%s", exc)
            break
        except Exception as exc:  # noqa: BLE001
            log.exception("%s 的理解階段失敗，繼續下一支：%s", vid, exc)

    log.info("understand 完成：%d/%d 支。%s", completed, len(targets), ledger.summary())
    return 0
