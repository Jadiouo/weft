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
    #: 攝影棚常態（中位幀）的指紋。**跨集比對用**——實測同一個播放清單的
    #: 第 14、27 集換了攝影棚背景（木質牆 vs 水墨山景），
    #: 「一個系列跑一次 S-1」的假設因此不成立（SDD v0.4 §4.0）。
    background_fingerprint: list[float] = field(default_factory=list)
    #: 這一支屬於哪個系列。**票 04 新增**：S-1 的比較基準改為
    #: 「本系列已跑過影片的彙總」，沒有這個欄位就分不出哪些是同系列的。
    #: `None` 表示單支處理，或**舊的 profile 檔**（那時還沒這個欄位）——
    #: 兩者都會被 `series_baseline` 排除在基準之外，而不是誤算進去。
    series_id: str | None = None
    #: 相異投影片數 ÷ 候選幀數。SDD §4.3b 要求納入 profile：
    #: 這個比例在同系列各集之間若大幅跳動，代表去重門檻不能通用。
    distinct_ratio: float | None = None

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class SeriesProfile:
    """系列層級的 profile 與不符判定。SDD §4.0 的輸出。"""

    series_id: str | None
    videos: list[VideoProfile] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)
    #: 不中止、但必須留下的觀察（例如換了攝影棚背景）。v0.4 §4.0 第 5 條。
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict:
        return {
            "series_id": self.series_id,
            "ok": self.ok,
            "mismatches": self.mismatches,
            "notes": self.notes,
            "videos": [v.to_dict() for v in self.videos],
        }


# --------------------------------------------------------------------------
# §4.0 的中止條件
#
# 這些數值是「與 §1.3 記載的 profile 顯著不符」的判準，不是 §5.2 的驗收
# 門檻——§5.2 量的是演算法準不準，這裡量的是**素材適不適用現有設計**。
# --------------------------------------------------------------------------

#: 1. 講者佔比的比較基準。**只用於記錄，不再中止**（v0.5 票 04）。
#:
#: §4.0 自己寫「這個階段只回答**這一支能不能用現有設計處理**」，
#: 但這條判準問的是「這一支像不像中醫講經第 1 集」——那是兩件事。
#: 純螢幕錄影（無講者入鏡）會嚴重偏離它，而那其實是**更好處理**的情況。
#: 機器人學課程十之八九會誤中止。
#:
#: **基準改為本系列已跑過影片的彙總**，只有系列第一支才退回 §1.3 的
#: 81.1%——那個數字是單支影片的觀察，v0.2 已把它降級為「範例，不是設計
#: 前提」，拿它當跨系列的通則正是這個 repo 犯過三次的同一個錯。
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
    #: 攝影棚常態的指紋——中位幀降到 8x8 後攤平。夠小可以進 JSON，
    #: 也夠粗不受壓縮雜訊影響，但換了背景會明顯不同（v0.4 §4.0）。
    fingerprint = cv2.resize(reference, (8, 8), interpolation=cv2.INTER_AREA).flatten()
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
        series_id=meta.series_id,
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
        background_fingerprint=[round(float(x), 4) for x in fingerprint],
    )


#: 與**所有已知**同系列 profile 的最小指紋距離超過此值，
#: 即視為「換了攝影棚背景」。
#:
#: 實測同一個播放清單的四集（水墨山景 ×2、木質牆 ×2）：
#:
#: | | 第1集 | 第5集 | 第14集 | 第27集 |
#: |---|---|---|---|---|
#: | 第1集 | — | 0.017 | 0.102 | 0.103 |
#: | 第5集 | 0.017 | — | 0.096 | 0.099 |
#: | 第14集 | 0.102 | 0.096 | — | 0.029 |
#:
#: 群內最大 0.029、跨群最小 0.096，**分離 3.3x**。0.06 取在兩者之間。
#: 餘裕不大（3.3x，勉強過 D1／R12 的 2x 判準），素材換系列時要重量。
#:
#: **這不是中止條件**——換背景是素材事實不是錯誤，但它代表 §4.3 的
#: 分界值要以本支自己的中位幀重算，不能沿用別支的。
BACKGROUND_DRIFT = 0.06


def nearest_background(profile: VideoProfile,
                       known: list[VideoProfile]) -> tuple[str, float] | None:
    """與**所有**已知 profile 比，回傳最近的那一支與距離。

    **不是「跟前一支比」**——「前一支」在檔案系統上取決於檔名排序，
    不是集數順序；而且系列可能在兩種背景之間來回切換。
    問「有沒有任何一支長得像我」才是對的問題。
    """
    best: tuple[str, float] | None = None
    for other in known:
        if other.video_id == profile.video_id:
            continue
        dist = background_distance(other, profile)
        if dist is None:
            continue
        if best is None or dist < best[1]:
            best = (other.video_id, dist)
    return best


def background_distance(a: VideoProfile, b: VideoProfile) -> float | None:
    """兩支影片的攝影棚常態差多遠。缺指紋時回傳 `None`。"""
    if not a.background_fingerprint or not b.background_fingerprint:
        return None
    if len(a.background_fingerprint) != len(b.background_fingerprint):
        return None
    return float(np.mean(np.abs(np.array(a.background_fingerprint)
                                - np.array(b.background_fingerprint))))


def background_notes(profiles: list[VideoProfile]) -> list[str]:
    """逐支與前一支比對背景。SDD v0.4 §4.0 的第 5 條判準。

    **不中止，但必須記錄**——實測第 14 集換了攝影棚背景，而前 13 集都沒換。
    把第 1 集的 profile 套到第 14 集，與 v0.1 把單支影片套到整個系列
    是同一個錯誤的不同尺度。
    """
    notes: list[str] = []
    for i, cur in enumerate(profiles):
        nearest = nearest_background(cur, profiles[:i])
        if nearest and nearest[1] >= BACKGROUND_DRIFT:
            notes.append(
                f"{cur.video_id} 的攝影棚背景與已知的每一支都不同"
                f"（最近的是 {nearest[0]}，差 {nearest[1]:.3f}，門檻 {BACKGROUND_DRIFT}）"
                f"——§4.3 的分界值須以本支自己的中位幀重算"
            )
    return notes


def check_mismatches(profiles: list[VideoProfile]) -> list[str]:
    """SDD §4.0 的中止條件。**只回傳真的處理不了的。**

    「偏離」與「處理不了」是兩件事（v0.5 票 04）。畫面結構與中醫講經
    不同不代表跑不動——偏離走 `deviation_notes`，只有這裡的才中止。
    """
    problems: list[str] = []
    if not profiles:
        return ["沒有任何影片可勘查"]

    # 判準 1（講者佔比）已移出中止條件，改為 `deviation_notes` 的記錄項。

    single_mode = [v for v in profiles if _is_single_mode(v)]
    two_mode = [v for v in profiles if not _is_single_mode(v)]
    if two_mode:
        worst = min(v.mode_separation for v in two_mode)
        if worst < MIN_MODE_SEPARATION:
            problems.append(
                f"模式分離度最低只有 {worst:.2f}x（要求 ≥{MIN_MODE_SEPARATION}x）"
                "——§4.3 步驟 2 的二分類在這個系列上不可靠"
            )
    elif single_mode:
        # **全片單一模式時分離度沒有意義**，不是不可靠。純螢幕錄影
        # （全程投影片）與純口播（全程講者）都會落在這裡，而兩者都比
        # 混合素材好處理。拿一個無意義的數字去中止是誤判。
        log.info("S-1：%d 支為單一畫面模式，跳過分離度判準（那個數字對它們沒有意義）",
                 len(single_mode))

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


#: 全螢幕佔比落在這個區間之外，視為「單一畫面模式」。
#: 純螢幕錄影趨近 1.0、純口播趨近 0.0，兩者的「模式分離度」都沒有意義。
#: 0.05 是保護性邊界不是校準值——實測樣本目前只有中醫講經（0.19–0.28）。
SINGLE_MODE_MARGIN = 0.05


def _is_single_mode(profile: VideoProfile) -> bool:
    return (profile.fullscreen_ratio < SINGLE_MODE_MARGIN
            or profile.fullscreen_ratio > 1.0 - SINGLE_MODE_MARGIN)


def series_baseline(out_dir, series_id: str | None) -> tuple[float, str]:
    """本系列已跑過影片的講者佔比彙總。回傳 `(基準值, 來源說明)`。

    **系列第一支才退回 §1.3。** 那個 81.1% 是單支影片的觀察，
    v0.2 已把它降級為「範例，不是設計前提」——拿它當跨系列的通則，
    與 v0.1 把單支推廣成系列通則是同一個錯。
    """
    known = [v for v in load_video_profiles(out_dir)
             if series_id and v.series_id == series_id]
    if not known:
        return SPEAKER_RATIO_REFERENCE, "§1.3（本系列尚無已跑過的影片）"
    ratio = statistics.mean(1.0 - v.fullscreen_ratio for v in known)
    return ratio, f"本系列已跑過的 {len(known)} 支彙總"


def deviation_notes(profiles: list[VideoProfile], baseline: float,
                    source: str) -> list[str]:
    """**偏離但不中止**的觀察。

    畫面結構與基準不同不代表處理不了。純螢幕錄影會嚴重偏離講者佔比，
    但它是更好處理的情況——把它算成中止條件會讓整個系列跑不起來。
    """
    if not profiles:
        return []
    ratio = statistics.mean(1.0 - v.fullscreen_ratio for v in profiles)
    if abs(ratio - baseline) <= SPEAKER_RATIO_TOLERANCE:
        return []
    direction = "更少" if ratio < baseline else "更多"
    return [
        f"講者佔比 {ratio:.1%}，偏離基準 {baseline:.1%}（{source}）"
        f"超過 ±{SPEAKER_RATIO_TOLERANCE:.0%}——畫面上的講者比基準{direction}。"
        f"**這不是中止條件**：v0.5 起主幹是逐字稿，投影片是輔助，"
        f"兩個方向的偏離都跑得動（票 04）。記下來是為了讓 §5.6 的抽檢知道"
        f"這批素材與既有黃金集不同型態。"
    ]


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

    baseline, source = series_baseline(cfg.out_dir, series_id)
    return SeriesProfile(
        series_id=series_id,
        videos=profiles,
        mismatches=check_mismatches(profiles),
        notes=background_notes(profiles) + deviation_notes(profiles, baseline, source),
    )


def write_video_profile(profile: VideoProfile, out_dir: Path) -> Path:
    """寫出 `out/profile/{video_id}.json`。SDD v0.4 §4.0——**逐支**。

    系列彙總（`{series_id}.json`）記的是「這個系列有多不齊」，
    不是拿來當通則的。逐支的這一份才是該支自己的量測結果。
    """
    import json

    target = out_dir / "profile"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{profile.video_id}.json"
    path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def load_video_profiles(out_dir: Path) -> list[VideoProfile]:
    """讀回既有的逐支 profile，供跨集比對用。"""
    import json

    target = out_dir / "profile"
    if not target.exists():
        return []
    out = []
    for path in sorted(target.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        # **用 `videos` 而不是 `series_id` 當判別式。** 票 04 把 `series_id`
        # 加進逐支 profile 之後，舊的判別式會把每一支都當成彙總檔排除掉——
        # 而症狀是「基準永遠退回 §1.3」，看起來像設定沒生效，不像讀檔壞了。
        if "video_id" not in data or "videos" in data:
            continue  # 系列彙總檔，不是逐支的
        known = {f for f in VideoProfile.__dataclass_fields__}
        # `series_id` 是票 04 才加的，舊的 profile 檔沒有這個鍵。
        # 缺的話補 None——那表示「不知道它屬於哪個系列」，
        # `series_baseline` 會把它排除在基準之外，而不是誤算進去。
        fields = {k: v for k, v in data.items() if k in known}
        fields.setdefault("series_id", None)
        out.append(VideoProfile(**fields))
    return out


def write_profile(profile: SeriesProfile, out_dir: Path) -> Path:
    """寫出 `out/profile/{series_id}.json`。SDD §4.0。"""
    import json

    target = out_dir / "profile"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{profile.series_id or 'single'}.json"
    path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
