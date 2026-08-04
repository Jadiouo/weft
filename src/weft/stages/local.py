"""S0–S3：本地管線，不花額度。SDD §4.1–§4.6，實作屬 Phase 1（§7.2）。

每個函式的簽名即為 SDD §4 的介面契約。Phase 1 只需把 `pending(...)` 換成
實作，不應改動簽名——簽名改了代表契約改了，那要先改 SDD。
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Config
from ..ir import CandidateSet, Lexicon, Segment, Slide, Transcript, VideoMeta
from ..paths import WorkPaths
from . import pending

log = logging.getLogger(__name__)


def s0_fetch(video_id: str, cfg: Config, work: WorkPaths) -> VideoMeta:
    """S0 取得。SDD §4.1。

    冪等鍵：video_id
    失敗行為：影片不可用（私人／刪除）→ 記入 skip list，繼續下一支
    """
    pending(
        "S0 取得",
        "§4.1",
        "Phase 1",
        [
            "yt-dlp 下載影片與字幕，字幕優先序：手動 > 自動 > 無",
            "確認下載的影片不含硬燒字幕（SDD §1.3 已驗證為播放器字幕）",
            "寫出 00_meta.json / 01_video.mp4 / 01_captions.vtt",
            "影片不可用時寫入 out/skiplist.json 並繼續，不中斷批次",
        ],
    )


def s1a_transcript(cfg: Config, work: WorkPaths, lexicon: Lexicon | None = None) -> Transcript:
    """S1a 逐字稿。SDD §4.2。

    冪等鍵：video_id + whisper_model + params_hash
    失敗行為：Whisper OOM → 降 batch size 重試一次 → 仍失敗標記 failed 並繼續
    """
    pending(
        "S1a 逐字稿",
        "§4.2",
        "Phase 1",
        [
            "有手動字幕 → 直接採用，source=manual_caption",
            "否則 faster-whisper large-v3（language=zh），source=whisper",
            "有自動字幕且也跑了 Whisper → 自動字幕存為 alt 供交叉檢查",
            "系列詞庫存在時餵入 initial_prompt（§9 文言文 ASR 的緩解）",
            "計算並寫入 raw_hash（§5.3 不變量 9 的比對基準）",
        ],
    )


def s1b_slides(cfg: Config, work: WorkPaths) -> tuple[CandidateSet, list[Slide]]:
    """S1b 投影片候選幀。SDD §4.3。移植 vid2slides 演算法，不 fork 該 repo。

    冪等鍵：video_id + fps + detector_params_hash
    失敗行為：偵測到 0 張投影片 → mode=transcript_only，不中斷（見 §4.3）
    """
    import shutil

    import cv2

    from ..ir import CandidateSet, FrameClass, FrameLabel, Slide, SlideCandidate
    from .detect import Run, detect_sections, drop_short_sections, split_runs
    from .frames import extract_frames, load_frames

    p = cfg.s1b
    fps = p.fps

    paths = extract_frames(work.video, work.frames_dir, fps)
    if not paths:
        raise RuntimeError(f"{work.video} 抽不出任何幀")

    frames = load_frames(paths, fps, p.downscale_short_side, p.blur_sigma, p.face_min_area_ratio)
    duration = len(frames) / fps

    # 每個 slide 段落內各自偵測換頁。speaker 段落不參與——SDD §4.3 把分類排在
    # 偵測之前是有原因的：實測顯示講者的輕微晃動（ink Jaccard 可達 0.4）比
    # 部分真實換頁還大，混在一起會淹掉訊號。
    sections = []
    for run in split_runs(frames):
        if run.is_speaker:
            continue
        found = detect_sections(
            frames, run, p.hmm_self_transition, p.min_ink_change,
            p.progressive_containment_ratio, p.progressive_merge,
        )
        sections += drop_short_sections(found, frames, p.min_slide_duration_sec, fps)

    half = 0.5 / fps  # 幀時間戳取的是每格中心，還原成格的邊界
    candidates: list[SlideCandidate] = []
    slides: list[Slide] = []
    if work.slides_dir.exists():
        shutil.rmtree(work.slides_dir)
    work.slides_dir.mkdir(parents=True, exist_ok=True)

    for i, section in enumerate(sections):
        t_start = max(0.0, frames[section.start].t - half)
        t_end = min(duration, frames[section.end - 1].t + half)
        key = frames[section.keyframe]
        build_times = [frames[b].t for b in section.build_indices]

        candidates.append(
            SlideCandidate(
                index=i,
                t_start=t_start,
                t_end=t_end,
                keyframe_t=key.t,
                build_frames=build_times if section.is_progressive else [],
            )
        )

        image = work.slide_image(i + 1)
        # 存原始解析度的那一幀，不是偵測用的 180px 縮圖——這張要送去 OCR 與 VLM
        img = cv2.imread(str(key.path))
        cv2.imwrite(str(image), img)
        slides.append(
            Slide(
                slide_id=f"slide_{i + 1:03d}",
                image_path=str(image.relative_to(work.dir)),
                t_first_seen=t_start,
                t_last_seen=t_end,
                is_progressive_final=section.is_progressive,
                build_frames=build_times if section.is_progressive else [],
            )
        )

    candidate_set = CandidateSet(
        video_id=work.video_id,
        fps=fps,
        duration=duration,
        frames=[
            FrameLabel(
                t=f.t,
                frame_class=FrameClass.SPEAKER if f.is_speaker else FrameClass.SLIDE,
                face_score=min(1.0, f.face_area_ratio),
                frame_path=str(f.path.relative_to(work.dir)) if f.path else None,
            )
            for f in frames
        ],
        candidates=candidates,
        params_hash=p.params_hash(),
    )
    work.candidates.write_text(candidate_set.model_dump_json(indent=2), encoding="utf-8")

    if not slides:
        log.warning("%s 偵測到 0 張投影片，將以 transcript_only 模式繼續（§4.3）", work.video_id)
    return candidate_set, slides


def s2_ocr(cfg: Config, work: WorkPaths, slides: list[Slide]) -> list[Slide]:
    """S2 OCR。SDD §4.4。PaddleOCR-VL，本地執行，無額度限制。

    冪等鍵：slide_set_hash + ocr_model
    """
    pending(
        "S2 OCR",
        "§4.4",
        "Phase 1",
        [
            "PaddleOCR-VL 對 03_slides/*.png 做 OCR，寫出 03_ocr.json",
            "直排文字的準確度未實測——此處不求完美，精確理解交給 S4 的 VLM",
        ],
    )


def s2b_lexicon(cfg: Config, work: WorkPaths, slides: list[Slide], series_id: str | None) -> Lexicon:
    """S2b 術語詞庫。SDD §4.4。scope 為**系列級**，不是單片級。"""
    pending(
        "S2b 術語詞庫",
        "§4.4",
        "Phase 1",
        [
            "從 OCR 文字抽候選術語（2–6 字專有名詞、四字詞、書名號與括號內文字）",
            "以 series_id 為 scope 累積，新影片的術語 append 進去",
            "寫出 04_lexicon.json",
        ],
    )


def s2c_correct(cfg: Config, work: WorkPaths, transcript: Transcript, lexicon: Lexicon) -> Transcript:
    """S2c 逐字稿術語校正。SDD §4.5。

    **這是本專案品質的關鍵路徑，不是可選功能。**

    失敗行為：詞庫為空 → 跳過，transcript_corrected = transcript_raw
    """
    pending(
        "S2c 逐字稿術語校正",
        "§4.5",
        "Phase 1",
        [
            "以拼音／字形相似度比對詞庫（中文 ASR 錯誤多為同音或近音）",
            "每次替換記錄 {from, to, source, method, score}",
            "只在時間鄰近的投影片詞庫中比對（±N 個 segment）",
            "transcript_raw 永不覆寫（§5.3 不變量 9）",
            "precision 門檻 0.90——寧可漏改，不可亂改",
        ],
    )


def s3_align(
    cfg: Config, work: WorkPaths, transcript: Transcript, candidates: CandidateSet
) -> list[Segment]:
    """S3 對齊。SDD §4.6。**這一步不呼叫 LLM**，避免與 S4 形成循環依賴。

    冪等鍵：transcript_hash + candidates_hash + embedding_model
    """
    pending(
        "S3 對齊",
        "§4.6",
        "Phase 1",
        [
            "以投影片切換時間戳粗切",
            "每個邊界取前後 ±20 秒的句子，算 embedding 與前後投影片 OCR 的相似度",
            "把邊界吸附到相似度轉折點，吸附範圍硬限制 ±20 秒",
            "純講者時段自成 segment，mode=speaker_only",
            "填入 cue_indices，使每句逐字稿恰好屬於一個 segment（§5.3 不變量 3）",
        ],
    )
