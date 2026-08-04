"""階段編排。SDD §6.3 續跑語意 + §6.4 producer/consumer 分離。

編排層負責的事，與各階段本身無關：
  - 依 state.json 判斷哪些階段可跳過（參數 hash 未變且已完成）
  - 參數 hash 變更時作廢下游（`VideoState.invalidate_downstream`）
  - 中途失敗時保留已完成的階段，讓下次執行接得上

實作屬 Phase 1（§7.2）。這裡先把控制流寫死，好讓「缺的是哪一階段」在
執行時就講清楚，而不是丟一個 ImportError。
"""

from __future__ import annotations

import logging

from .config import Config
from .paths import WorkPaths
from .state import Stage, VideoState

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


def resolve_targets(target: str) -> list[str]:
    """把 playlist URL / video URL / video_id 解析成 video_id 清單。"""
    from .stages import pending

    pending(
        "target 解析",
        "§4.1",
        "Phase 1",
        [
            "用 yt-dlp 展開 playlist 為 video_id 清單，保留 episode_index",
            "單支影片 URL 或裸 video_id 直接回傳單元素清單",
        ],
    )


def run_prepare(target: str, cfg: Config, force: bool = False) -> int:
    """S0–S3。不受額度限制，可一次跑完整個系列。SDD §6.4。"""
    from .stages import local

    video_ids = resolve_targets(target)
    for video_id in video_ids:
        work = WorkPaths(cfg.work_dir, video_id)
        work.ensure_dirs()
        state = VideoState.load_or_new(work.state, video_id)
        if force:
            state = VideoState(video_id=video_id)
        sync_state(cfg, state)

        local.s0_fetch(video_id, cfg, work)  # 之後接續 S1a…S3
    return 0


def run_understand(cfg: Config, video_id: str | None = None, max_requests: int | None = None) -> int:
    """S4–S6。消化 prepare 囤下的 buffer，額度耗盡即停。SDD §6.4。"""
    from .stages import cloud

    cloud.s4_understand(cfg, WorkPaths(cfg.work_dir, video_id or ""), [], None)  # type: ignore[arg-type]
    return 0
