"""階段編排。SDD §6.3 續跑語意 + §6.4 producer/consumer 分離。

編排層負責的事，與各階段本身無關：
  - 依 state.json 判斷哪些階段可跳過（參數 hash 未變且已完成）
  - 參數 hash 變更時作廢下游（`VideoState.invalidate_downstream`）
  - 中途失敗時保留已完成的階段，讓下次執行接得上
"""

from __future__ import annotations

import contextlib

import json
import logging

from .config import Config
from .ir import CandidateSet, Slide, Transcript, VideoMeta
from .paths import OutPaths, WorkPaths
from .stages import dedup
from .stages.survey import BACKGROUND_DRIFT
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
        Stage.S1C_DEDUP: cfg.s1c,
        Stage.S3_ALIGN: cfg.s3,
        Stage.S4A_SLIDES: cfg.s4a,
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


@contextlib.contextmanager
def _stage(stage: "Stage"):
    """把階段內拋出的例外包成 `StageFailure`，記下是哪一個階段。

    `VideoUnavailable` 不包——`run_prepare` 對它有專門的處置
    （記進 skip list 而不是算成失敗，§4.1）。
    """
    from .stages.fetch import VideoUnavailable

    try:
        yield
    except (VideoUnavailable, StageFailure):
        raise
    except Exception as exc:
        raise StageFailure(stage, exc) from exc


class StageFailure(Exception):
    """帶著**真正失敗的那個階段**往上拋。

    原本 `run_prepare` 的例外處理寫死 `state.mark_failed(Stage.S0_FETCH, ...)`，
    所以不管哪個階段爆掉都記成「S0 取得失敗」。實測 Whisper 的
    `CUDA failed with error out of memory`（S1a）被記成 S0——
    查問題的人會去看下載，而下載完全正常。

    **錯誤訊息指錯地方比沒有錯誤訊息更糟**：它會主動把人帶往錯的方向
    （D29 的行號 bug 是同一類）。
    """

    def __init__(self, stage: "Stage", cause: BaseException):
        super().__init__(f"{stage.value}：{cause}")
        self.stage = stage
        self.cause = cause


def prepare_one(
    video_id: str,
    cfg: Config,
    series_id: str | None = None,
    episode_index: int | None = None,
    force: bool = False,
) -> VideoState:
    """跑單支影片的 S0–S3。每個階段各自檢查是否可跳過（§6.3）。

    任一階段失敗時包成 `StageFailure` 往上拋，**帶著那個階段的身分**。
    """
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
        with _stage(Stage.S0_FETCH):
            meta = local.s0_fetch(video_id, cfg, work)
        if series_id or episode_index:
            meta = meta.model_copy(update={"series_id": series_id, "episode_index": episode_index})
            work.meta.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        state.mark_done(Stage.S0_FETCH, stage_params(cfg, Stage.S0_FETCH))
        state.save(work.state)

    # ---- S-1 素材勘查（v0.4：**逐支跑**，§4.0）----
    # 與 S1b 共用抽幀，邊際成本接近零。實測同一個播放清單的第 14、27 集
    # 換了攝影棚背景，「一個系列跑一次」的假設不成立。
    if cfg.survey_each_video:
        from .stages.survey import (
            load_video_profiles,
            nearest_background,
            profile_video,
            write_video_profile,
        )

        prof_path = cfg.out_dir / "profile" / f"{work.video_id}.json"
        if not prof_path.exists():
            profile = profile_video(work.video_id, work, cfg)
            nearest = nearest_background(profile, load_video_profiles(cfg.out_dir))
            if nearest and nearest[1] >= BACKGROUND_DRIFT:
                # **不中止**——換背景是素材事實不是錯誤。但要留下記錄，
                # 因為 §4.3 的分界值必須以本支自己的中位幀重算。
                log.warning("S-1 %s：攝影棚背景與已知的每一支都不同"
                            "（最近的是 %s，差 %.3f，門檻 %.2f）——"
                            "分界值以本支自己的中位幀重算",
                            work.video_id, nearest[0], nearest[1], BACKGROUND_DRIFT)
            write_video_profile(profile, cfg.out_dir)
            log.info("S-1 %s：全螢幕 %.1f%%、分離度 %.2fx、%d 個區段",
                     work.video_id, profile.fullscreen_ratio * 100,
                     profile.mode_separation, profile.section_count)

    # ---- S1b 靜止區段 ----
    if satisfied(Stage.S1B_SLIDES) and work.candidates.exists():
        candidates = CandidateSet.model_validate_json(work.candidates.read_text(encoding="utf-8"))
        slides = _slides_from(work, candidates)
    else:
        with _stage(Stage.S1B_SLIDES):
            candidates, slides = local.s1b_slides(cfg, work)
        state.mark_done(Stage.S1B_SLIDES, stage_params(cfg, Stage.S1B_SLIDES))
        state.save(work.state)

    # ---- S1c 投影片去重（§4.3b，純本地）----
    if satisfied(Stage.S1C_DEDUP) and work.dedup.exists():
        _apply_dedup(work, slides)
    else:
        with _stage(Stage.S1C_DEDUP):
            stats = dedup.s1c_dedup(cfg, work, slides, candidates)
        work.dedup.write_text(json.dumps(
            {"stats": stats,
             "slides": {s.slide_id: {"duplicate_of": s.duplicate_of,
                                     "occurrences": [list(o) for o in s.occurrences]}
                        for s in slides}},
            ensure_ascii=False, indent=1), encoding="utf-8")
        state.mark_done(Stage.S1C_DEDUP, stage_params(cfg, Stage.S1C_DEDUP))
        state.save(work.state)

    # ---- S1a 逐字稿 ----
    if satisfied(Stage.S1A_TRANSCRIPT) and work.transcript.exists():
        transcript = Transcript.model_validate_json(work.transcript.read_text(encoding="utf-8"))
    else:
        with _stage(Stage.S1A_TRANSCRIPT):
            transcript = local.s1a_transcript(cfg, work)
        state.mark_done(Stage.S1A_TRANSCRIPT, stage_params(cfg, Stage.S1A_TRANSCRIPT))
        state.save(work.state)

    # ---- S3 對齊 ----
    if not satisfied(Stage.S3_ALIGN):
        with _stage(Stage.S3_ALIGN):
            local.s3_align(cfg, work, transcript, candidates)
        state.mark_done(Stage.S3_ALIGN, stage_params(cfg, Stage.S3_ALIGN))
        state.save(work.state)

    return state


def _apply_dedup(work: WorkPaths, slides: list[Slide]) -> None:
    """把落地的 S1c 結果套回 Slide 物件（續跑用）。

    去重是**衍生狀態**，續跑時必須重建而不是沿用預設值——D22 記的教訓：
    只在「新計算」那條路上做的話，續跑會拿到未去重的 slides。
    """
    if not work.dedup.exists():
        return
    data = json.loads(work.dedup.read_text(encoding="utf-8")).get("slides", {})
    for slide in slides:
        row = data.get(slide.slide_id)
        if not row:
            continue
        slide.duplicate_of = row.get("duplicate_of")
        slide.occurrences = [tuple(o) for o in row.get("occurrences") or []]


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
            # **記在真正失敗的那個階段上。** 原本寫死 S0，於是 Whisper 的
            # CUDA OOM 被記成「下載失敗」——查問題的人會去看下載，而下載
            # 完全正常。認不出階段時才退回 S0，並在訊息裡說明。
            stage = exc.stage if isinstance(exc, StageFailure) else Stage.S0_FETCH
            detail = str(exc.cause) if isinstance(exc, StageFailure) else str(exc)
            log.exception("%s 在 %s 失敗，繼續下一支：%s", video_id, stage.value, detail)
            work = WorkPaths(cfg.work_dir, video_id)
            state = VideoState.load_or_new(work.state, video_id)
            state.mark_failed(stage, detail)
            state.save(work.state)

    if failed:
        # **有失敗就要在最後一行看得到。** 批次跑數十支時，中途的
        # log.exception 早就被沖掉了，而收尾那一行是唯一會被讀的東西。
        log.error("prepare 完成：%d 支目標，**%d 支失敗**（見各支的 state.json）",
                  len(targets), failed)
    else:
        log.info("prepare 完成：%d 支目標，全部成功", len(targets))
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
    _apply_dedup(work, slides)   # 去重是衍生狀態，續跑必須重建（D22）
    transcript = Transcript.model_validate_json(work.transcript.read_text(encoding="utf-8"))

    # ---- S4a 投影片理解（逐張相異投影片，§4.7a）----
    from .stages import slides as slides_stage
    from .stages.dedup import representatives
    from .stages.providers import costs_quota

    need_s4a = not state.is_satisfied(Stage.S4A_SLIDES, stage_params(cfg, Stage.S4A_SLIDES))
    if not need_s4a:
        # 續跑：從快取重建投影片文字（D22——衍生狀態不能只在新算的那條路上做）
        recovered = slides_stage.rehydrate(cfg, work, slides)
        expected = len(representatives(slides))
        if recovered < expected:
            # **重建不出來就不能假裝沒事。** state 說 S4a 完成，但快取檔不見了
            # （手動刪掉、磁碟壞了）——這時若默默往下走，所有投影片都會變成
            # 「沒有文字」，管線照樣產出一份少了一半內容的知識庫，
            # 而每一項機械檢查都是綠的。
            log.warning("S4a 的快取只重建出 %d/%d 張，就地重跑 S4a", recovered, expected)
            state.mark_stale(Stage.S4A_SLIDES)
            need_s4a = True

    if need_s4a:
        from .quota import QuotaExhausted, QuotaLedger

        ledger = QuotaLedger(out.quota_db, cfg.quota)

        def _record(spec, in_tok, out_tok, sid, status):
            # 本地模型不佔額度——§6.5 的估算依實際配置而定
            if costs_quota(spec):
                ledger.record(spec, in_tok, out_tok, sid, status)

        if costs_quota(cfg.s4a.model):
            try:
                ledger.check(planned_requests=1, model=cfg.s4a.model)
            except QuotaExhausted as exc:
                log.warning("S4a %s；停止本日處理，進度已保存（§6.1）", exc)
                state.save(work.state)
                return False

        slides_stage.s4a_understand_slides(cfg, work, slides, on_call=_record)
        state.mark_done(Stage.S4A_SLIDES, stage_params(cfg, Stage.S4A_SLIDES))
        state.save(work.state)

    # ---- S4c 逐段理解 ----
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

    # ---- S4b 詞庫事後校正（純本地，不花額度）----
    # **必須在 S4 全部跑完之後**：詞庫要從所有投影片建起來，
    # 第 5 張投影片上的術語可能在第 20 段被聽錯（D25）。
    from .stages import lexicon as lexicon_stage

    lexicon_stage.apply_to_video(segments, slides, transcript, cfg.s4b)

    # S4／S4b 改過逐字稿的 text_corrected（術語校正），落地
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

    log.info("understand：%d 支待處理。%s", len(targets), ledger.summary(cfg.s4.model))
    completed = 0
    for vid in targets:
        if max_requests is not None and ledger.usage_today().requests >= max_requests:
            log.info("已達本次執行的請求上限 %d，停止", max_requests)
            break
        try:
            if understand_one(vid, cfg):
                completed += 1
            else:
                # 未跑完。若是額度造成的，後面的影片同樣跑不動，不必逐支重試；
                # 全本地配置也可能未跑完（例如某幾段模型沒回），那與額度無關，
                # **不要誤報成額度用盡**——訊息錯了會讓人往錯的方向查。
                if ledger.remaining() <= 0:
                    log.info("額度用盡，停止本日處理。下次重置：%s",
                             ledger.next_reset().isoformat())
                else:
                    log.warning("%s 未完整跑完（與額度無關，剩餘 %d 次）。"
                                "進度已保存，可直接續跑", vid, ledger.remaining())
                    continue
                break
        except ApiKeyMissing as exc:
            log.error("%s", exc)
            return 2
        except QuotaExhausted as exc:
            log.info("%s", exc)
            break
        except Exception as exc:  # noqa: BLE001
            log.exception("%s 的理解階段失敗，繼續下一支：%s", vid, exc)

    log.info("understand 完成：%d/%d 支。%s", completed, len(targets), ledger.summary(cfg.s4.model))
    return 0
