"""S-1 素材勘查。SDD §4.0。

**每個新系列開跑前執行一次，不是每支影片都跑。**

存在的理由（SDD §4.0 原文）：v0.1 的 §1.3 把單支影片的觀察當作系列通則，
其中兩項描述與實際素材不符。錯誤一路傳導到 S1b 的分類設計，最終產品的
chunk 內容是攝影棚背板的裝飾字——而 §5.3 的十條不變量、§5.4 的溯源檢查
**全部通過**。機械檢查抓不到「素材與假設不符」這類錯誤，因為每一步都
忠實執行了錯誤的假設。

S-1 的職責就是在第一天把這件事抓出來。**它不產生知識庫內容**，只回答
「這個系列能不能用現有設計處理」。
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: 分類用的縮圖。夠小才能把整支影片一次載進記憶體算中位幀。
TINY = (64, 36)


@dataclass
class VideoProfile:
    """單支影片的勘查結果。"""

    video_id: str
    duration: float
    frame_count: int
    fullscreen_ratio: float
    mode_separation: float
    otsu_cut: float
    section_count: int
    sections_per_minute: float
    dwell_median: float
    dwell_min: float
    dwell_max: float
    transition_frames: int
    transition_ratio: float
    camera_motion_frames: int
    has_manual_caption: bool
    has_auto_caption: bool

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class SeriesProfile:
    """系列層級的 profile 與不符判定。SDD §4.0 的輸出。"""

    series_id: str | None
    videos: list[VideoProfile] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict:
        return {
            "series_id": self.series_id,
            "ok": self.ok,
            "mismatches": self.mismatches,
            "videos": [v.to_dict() for v in self.videos],
        }


# --------------------------------------------------------------------------
# §4.0 的中止條件
#
# 這些數值是「與 §1.3 記載的 profile 顯著不符」的判準，不是 §5.2 的驗收
# 門檻——§5.2 量的是演算法準不準，這裡量的是**素材適不適用現有設計**。
# --------------------------------------------------------------------------

#: 1. 講者佔比偏離 §1.3 記載值（81.1%）超過此幅度
SPEAKER_RATIO_REFERENCE = 0.811
SPEAKER_RATIO_TOLERANCE = 0.20

#: 2. 模式分離度低於此值代表二分類不可靠（§4.3 步驟 2 實測為 5.07x）
#:
#: **量測時必須排除轉場帶。** 交叉淡化幀本質上介於兩類之間，把它們算進去
#: 會讓任何有轉場的影片分離度趨近 1——實測 zIglvjoU9vo 含轉場時是 1.05x，
#: 排除後是 5.07x。所以用穩健分位數（攝影棚 p99 vs 全螢幕 p1）而非極值。
MIN_MODE_SEPARATION = 3.0
SEPARATION_PERCENTILE = 99.0

#: 3. 區段密度（每分鐘幾個靜止區段）。這直接決定 VLM 的請求數：
#: SDD §6.5 的預算是 42 分鐘影片約 14 次請求，批次 3 張即約 42 個區段
#: ≈ 1 個/分鐘。超過此上限代表過度切分，額度會爆。
#:
#: v0.2 原本檢查「投影片停留時間中位數 ≥10 秒」，但 v0.3 的區段包含講者
#: 段落，與 §5.1 的「投影片停留 30–120 秒」不是同一回事——區段密度才是
#: v0.3 架構下真正可行動的指標。
MAX_SECTIONS_PER_MINUTE = 3.0

#: 4. 偵測到的畫面模式超過兩種。以「分布的雙峰性」近似——
#: 若中間帶（介於兩群之間）的幀太多，代表不只兩種模式。
MAX_MIDBAND_RATIO = 0.15


def _otsu(values: np.ndarray) -> float:
    span = float(values.max())
    if span <= 0:
        return 0.0
    u8 = np.clip(values / span * 255.0, 0, 255).astype(np.uint8)
    level, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(level) / 255.0 * span


def profile_video(video_id: str, work, cfg) -> VideoProfile:
    """量測單支影片。SDD §4.0 的六項必測。"""
    from ..ir import VideoMeta
    from .detect import detect_sections, drop_short_sections
    from .frames import extract_frames, load_frames

    p = cfg.s1b
    meta = VideoMeta.model_validate_json(work.meta.read_text(encoding="utf-8"))
    paths = extract_frames(work.video, work.frames_dir, p.fps)
    if not paths:
        raise RuntimeError(f"{work.video} 抽不出任何幀")

    # --- 畫面模式分布（與 §4.3 步驟 2 同手法）---
    tiny = np.stack([
        cv2.resize(cv2.imread(str(x), cv2.IMREAD_GRAYSCALE), TINY,
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        for x in paths
    ])
    reference = np.median(tiny, axis=0)
    distance = np.mean(np.abs(tiny - reference), axis=(1, 2))
    cut = _otsu(distance)
    is_fullscreen = distance > cut

    # --- 模式分離度（排除轉場帶）---
    # 用穩健分位數而非極值：轉場幀介於兩類之間，用 min/max 會讓任何有
    # 交叉淡化的影片分離度趨近 1，量到的是轉場而不是模式的可分性。
    low, high = distance[~is_fullscreen], distance[is_fullscreen]
    if low.size and high.size:
        low_edge = float(np.percentile(low, SEPARATION_PERCENTILE))
        high_edge = float(np.percentile(high, 100.0 - SEPARATION_PERCENTILE))
        separation = high_edge / low_edge if low_edge > 0 else 0.0
    else:
        separation = 0.0

    # --- 轉場型態：介於兩群之間的幀 ---
    # 轉場幀是兩個畫面的混合，距離會落在兩群中間。用「高於攝影棚群的
    # 99 分位，但低於門檻」界定。
    if low.size:
        midband_floor = float(np.percentile(low, 99))
        transition = int(((distance > midband_floor) & (~is_fullscreen)).sum())
    else:
        midband_floor, transition = 0.0, 0

    # --- 鏡頭運動：攝影棚幀內部的畫面位移 ---
    # 推近會讓整幀相對基準位移，但**不會**讓它跨過模式門檻（實測餘裕
    # 8–10 倍）。所以量的是「攝影棚群內部的離群值」。
    studio = distance[~is_fullscreen]
    if studio.size > 10:
        median = float(np.median(studio))
        mad = float(np.median(np.abs(studio - median))) or 1e-6
        camera_motion = int((((studio - median) / mad) > 8.0).sum())
    else:
        camera_motion = 0

    # --- 停留時間分布：跑一次靜止區段偵測 ---
    frames = load_frames(paths, p.fps, p.downscale_short_side, p.blur_sigma)
    sections = detect_sections(
        frames, p.hmm_self_transition, p.min_ink_change,
        p.progressive_containment_ratio, p.progressive_merge,
    )
    sections = drop_short_sections(sections, frames, p.min_slide_duration_sec, p.fps)
    dwell = [(s.end - s.start) / p.fps for s in sections] or [0.0]

    return VideoProfile(
        video_id=video_id,
        duration=len(paths) / p.fps,
        frame_count=len(paths),
        fullscreen_ratio=float(is_fullscreen.mean()),
        mode_separation=separation,
        otsu_cut=float(cut),
        section_count=len(sections),
        sections_per_minute=len(sections) / max(1e-6, len(paths) / p.fps / 60.0),
        dwell_median=float(statistics.median(dwell)),
        dwell_min=float(min(dwell)),
        dwell_max=float(max(dwell)),
        transition_frames=transition,
        transition_ratio=transition / len(paths),
        camera_motion_frames=camera_motion,
        has_manual_caption=meta.has_manual_caption,
        has_auto_caption=meta.has_auto_caption,
    )


def check_mismatches(profiles: list[VideoProfile]) -> list[str]:
    """SDD §4.0 的四條中止條件。"""
    problems: list[str] = []
    if not profiles:
        return ["沒有任何影片可勘查"]

    speaker_ratio = statistics.mean(1.0 - v.fullscreen_ratio for v in profiles)
    if abs(speaker_ratio - SPEAKER_RATIO_REFERENCE) > SPEAKER_RATIO_TOLERANCE:
        problems.append(
            f"講者佔比 {speaker_ratio:.1%} 偏離 §1.3 記載的 "
            f"{SPEAKER_RATIO_REFERENCE:.1%} 超過 ±{SPEAKER_RATIO_TOLERANCE:.0%}"
            "——這個系列的畫面結構與現有設計的假設不同"
        )

    worst = min(v.mode_separation for v in profiles)
    if worst < MIN_MODE_SEPARATION:
        problems.append(
            f"模式分離度最低只有 {worst:.2f}x（要求 ≥{MIN_MODE_SEPARATION}x）"
            "——§4.3 步驟 2 的二分類在這個系列上不可靠"
        )

    density = max(v.sections_per_minute for v in profiles)
    if density > MAX_SECTIONS_PER_MINUTE:
        worst_video = max(profiles, key=lambda v: v.sections_per_minute)
        estimated = worst_video.section_count / 3  # 批次 3 張/次
        problems.append(
            f"區段密度最高 {density:.1f} 個/分鐘（上限 {MAX_SECTIONS_PER_MINUTE}）"
            f"——{worst_video.video_id} 會需要約 {estimated:.0f} 次 VLM 請求，"
            "遠超 §6.5 的預算。多半是過度切分"
        )

    midband = statistics.mean(v.transition_ratio for v in profiles)
    if midband > MAX_MIDBAND_RATIO:
        problems.append(
            f"介於兩種模式之間的幀佔 {midband:.1%}（上限 {MAX_MIDBAND_RATIO:.0%}）"
            "——畫面模式可能不只兩種，§1.3 的二分假設不成立"
        )

    return problems


def survey(video_ids: list[str], cfg, series_id: str | None = None) -> SeriesProfile:
    """跑完一個系列的勘查。SDD §4.0。"""
    from ..paths import WorkPaths

    profiles: list[VideoProfile] = []
    for video_id in video_ids:
        work = WorkPaths(cfg.work_dir, video_id)
        if not work.video.exists():
            log.warning("%s 的影片不在 work/，跳過（請先 weft prepare）", video_id)
            continue
        profiles.append(profile_video(video_id, work, cfg))
        log.info("S-1 %s：全螢幕 %.1f%%，分離度 %.2fx，%d 個區段"
                 "（%.1f 個/分鐘，約 %d 次 VLM 請求）",
                 video_id, profiles[-1].fullscreen_ratio * 100,
                 profiles[-1].mode_separation, profiles[-1].section_count,
                 profiles[-1].sections_per_minute,
                 -(-profiles[-1].section_count // 3))

    return SeriesProfile(
        series_id=series_id,
        videos=profiles,
        mismatches=check_mismatches(profiles),
    )


def write_profile(profile: SeriesProfile, out_dir: Path) -> Path:
    """寫出 `out/profile/{series_id}.json`。SDD §4.0。"""
    import json

    target = out_dir / "profile"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{profile.series_id or 'single'}.json"
    path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
