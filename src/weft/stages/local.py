"""S0–S3：本地管線，不花額度。SDD §4.1–§4.6，實作屬 Phase 1（§7.2）。

每個函式的簽名即為 SDD §4 的介面契約。Phase 1 只需把 `pending(...)` 換成
實作，不應改動簽名——簽名改了代表契約改了，那要先改 SDD。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import Config
from ..ir import CandidateSet, Segment, Slide, Transcript, VideoMeta
from ..paths import WorkPaths
from . import pending

log = logging.getLogger(__name__)


def s0_fetch(video_id: str, cfg: Config, work: WorkPaths) -> VideoMeta:
    """S0 取得。SDD §4.1。

    冪等鍵：video_id
    失敗行為：影片不可用（私人／刪除）→ 記入 skip list，繼續下一支
    """
    from ..ir import VideoMeta
    from .fetch import download

    work.ensure_dirs()
    info = download(video_id, work, cfg)

    langs = cfg.s0.caption_langs
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    caption_lang = next(
        (lang for lang in langs if lang in manual),
        next((lang for lang in langs if lang in auto), None),
    )

    meta = VideoMeta(
        video_id=video_id,
        title=info.get("title") or video_id,
        duration=float(info.get("duration") or 0.0),
        url=f"https://www.youtube.com/watch?v={video_id}",
        upload_date=info.get("upload_date"),
        has_manual_caption=any(lang in manual for lang in langs),
        has_auto_caption=any(lang in auto for lang in langs),
        caption_lang=caption_lang,
        video_path=str(work.video.relative_to(work.dir)),
        caption_path=str(work.captions.relative_to(work.dir)) if work.captions.exists() else None,
    )
    if meta.duration <= 0:
        raise RuntimeError(f"{video_id} 的長度為 {meta.duration}——metadata 不可信，中止")

    work.meta.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    log.info(
        "S0 %s：%s（%.0fs，字幕 %s）",
        video_id, meta.title, meta.duration,
        f"{caption_lang}{'（手動）' if meta.has_manual_caption else '（自動）'}" if caption_lang else "無",
    )
    return meta


def s1a_transcript(cfg: Config, work: WorkPaths) -> Transcript:
    """S1a 逐字稿。SDD §4.2。

    冪等鍵：video_id + whisper_model + params_hash
    失敗行為：Whisper OOM → 降 batch size 重試一次 → 仍失敗標記 failed 並繼續
    """
    from ..ir import Transcript, TranscriptCue, TranscriptSource, VideoMeta
    from .transcribe import parse_vtt, to_traditional, whisper_transcribe

    meta = VideoMeta.model_validate_json(work.meta.read_text(encoding="utf-8"))
    p = cfg.s1a

    def to_cues(rows: list[tuple[float, float, str]]) -> list[TranscriptCue]:
        return [
            TranscriptCue(index=i, t_start=a, t_end=max(b, a + 0.01), text_raw=t, text_corrected=t)
            for i, (a, b, t) in enumerate(rows)
        ]

    alt_cues = None
    converted = None
    if meta.has_manual_caption and work.captions.exists():
        # 策略 1：手動字幕是人打的，品質遠高於任何 ASR
        # **不做簡繁轉換**——手動字幕的字集是作者的選擇（D24）。
        rows = parse_vtt(work.captions)
        source, model = TranscriptSource.MANUAL_CAPTION, None
        log.info("S1a %s：採用手動字幕，%d 句", work.video_id, len(rows))
    else:
        # 策略 2／3：跑 Whisper；若另有自動字幕則一併保留供交叉檢查
        #
        # v0.3 拿掉了 initial_prompt 的系列詞庫——詞庫來自本地 OCR，已移除。
        # 文言文 ASR 的錯字改由 S4 對照投影片修正（known-risks R11）。
        rows = whisper_transcribe(work.video, cfg, None)
        source, model = TranscriptSource.WHISPER, p.whisper_model
        # ASR 的字集是模型產物，統一轉成繁體再往下走（D24／R18）
        rows = to_traditional(rows, p.asr_script_conversion)
        converted = p.asr_script_conversion
        if meta.has_auto_caption and work.captions.exists():
            # 自動字幕同樣是 ASR，一併轉
            alt_cues = to_cues(to_traditional(parse_vtt(work.captions),
                                              p.asr_script_conversion))
        log.info("S1a %s：Whisper %s，%d 句%s", work.video_id, p.whisper_model,
                 len(rows), f"（已轉繁 {converted}）" if converted else "")

    if not rows:
        raise RuntimeError(f"{work.video_id} 產生不出任何逐字稿")

    cues = to_cues(rows)
    transcript = Transcript(
        video_id=work.video_id,
        source=source,
        language=p.language,
        cues=cues,
        raw_hash=Transcript.compute_raw_hash(cues),
        alt_cues=alt_cues,
        model=model,
        params_hash=p.params_hash(),
        script_conversion=converted,
    )
    work.transcript.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return transcript


def s1b_slides(cfg: Config, work: WorkPaths) -> tuple[CandidateSet, list[Slide]]:
    """S1b 靜止區段偵測。SDD §4.3。

    **v0.3 起 CV 只負責去重。** 它找出畫面靜止的區段、每段取一張代表幀，
    **不判斷那是投影片還是講者**——那由 S4 的 VLM 回答（`is_slide`）。

    移除分類的理由（實測，見 docs/decisions.md D16）：
      - 講者人臉面積佔比 0.005–0.026，門檻 0.04，偵測不到
      - 投影片中的人像（胚胎圖）被誤測成 0.024，與講者畫面完全重疊
      - 而 VLM 本來就要看這張圖，多回一個布林值幾乎免費

    冪等鍵：video_id + fps + detector_params_hash
    失敗行為：偵測到 0 個區段 → 全片 transcript_only，不中斷（見 §4.3）
    """
    import shutil

    import cv2

    from ..ir import CandidateSet, FrameLabel, Slide, SlideCandidate
    from .detect import detect_sections, drop_short_sections
    from .frames import extract_frames, load_frames

    p = cfg.s1b
    fps = p.fps

    paths = extract_frames(work.video, work.frames_dir, fps)
    if not paths:
        raise RuntimeError(f"{work.video} 抽不出任何幀")

    frames = load_frames(paths, fps, p.downscale_short_side, p.blur_sigma)
    duration = len(frames) / fps

    # 全片一視同仁。講者段落會自然形成很長的靜止區段（攝影棚機位固定，
    # ink 遮罩幾乎不變），VLM 看到代表幀就會回 is_slide: false。
    sections = detect_sections(
        frames, p.hmm_self_transition, p.min_ink_change,
        p.progressive_containment_ratio, p.progressive_merge,
    )
    sections = drop_short_sections(sections, frames, p.min_slide_duration_sec, fps)

    half = 0.5 / fps  # 幀時間戳取的是每格中心，還原成格的邊界
    candidates: list[SlideCandidate] = []
    slides: list[Slide] = []
    if work.slides_dir.exists():
        shutil.rmtree(work.slides_dir)
    work.slides_dir.mkdir(parents=True, exist_ok=True)

    for i, section in enumerate(sections):
        t_start = max(0.0, frames[section.start].t - half)
        t_end = min(duration, frames[section.end - 1].t + half)
        key = frames[section.keyframe(frames)]
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
        # 存原始解析度的那一幀，不是偵測用的 180px 縮圖——這張要送去 VLM
        cv2.imwrite(str(image), cv2.imread(str(key.path)))
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
                frame_path=str(f.path.relative_to(work.dir)) if f.path else None,
            )
            for f in frames
        ],
        candidates=candidates,
        params_hash=p.params_hash(),
    )
    work.candidates.write_text(candidate_set.model_dump_json(indent=2), encoding="utf-8")

    log.info("S1b %s：%d 幀 → %d 個靜止區段（%d 個為逐條動畫合併）",
             work.video_id, len(frames), len(slides),
             sum(1 for s in sections if s.is_progressive))
    if not slides:
        log.warning("%s 偵測到 0 個區段，將以 transcript_only 模式繼續（§4.3）", work.video_id)
    return candidate_set, slides


def s3_align(
    cfg: Config, work: WorkPaths, transcript: Transcript, candidates: CandidateSet
) -> list[Segment]:
    """S3 分段。SDD §4.6（v0.5 改寫）。**這一步不呼叫 LLM**。

    **主幹是逐字稿，不是投影片切換。** v0.4 以前由切換決定分段，
    那不是準不準的問題而是結構性的：講者對著同一張圖講完兩件事時，
    它一刀都不會切。票 07 實測保留集召回只有 0.60，純語意是 0.93。

    投影片降級為「這一段螢幕上是哪一張」的註記（`topic_windows`），
    **不參與決定切在哪裡**。

    `mode` 在此只是**暫定**——有候選幀就先標 slide，S4 判定 `is_slide=false`
    時會降級為 speaker_only 並清掉 slide_ref。

    冪等鍵：transcript_hash + candidates_hash + s3_params_hash
    """
    from ..ir import BoundaryMethod, Segment, SegmentMode
    from .align import assign_cues, coarse_windows, topic_windows
    from .segment import enforce_min_length, topic_boundaries

    p = cfg.s3
    if p.method == "slide":
        # v0.4 的行為，保留是為了能做對照，**不是建議值**
        windows = coarse_windows(candidates.candidates, candidates.duration,
                                 p.min_segment_sec)
        method = BoundaryMethod.SLIDE_SWITCH
    else:
        cuts = enforce_min_length(
            topic_boundaries(transcript.cues, p.block_chars, p.block_window,
                             p.depth_alpha),
            candidates.duration, p.min_segment_sec)
        windows = topic_windows(cuts, candidates.duration, candidates.candidates)
        method = BoundaryMethod.TOPIC_SHIFT
        log.info("S3 %s：逐字稿切出 %d 個話題邊界（方法 %s，偽句 %d 字）",
                 work.video_id, len(cuts), p.method, p.block_chars)
    buckets = assign_cues(windows, transcript.cues)
    by_index = {c.index: c for c in transcript.cues}

    segments: list[Segment] = []
    for i, (w, cue_indices) in enumerate(zip(windows, buckets)):
        picked = [by_index[j] for j in cue_indices]
        if w.slide_id is None:
            mode = SegmentMode.SPEAKER_ONLY if candidates.candidates else SegmentMode.TRANSCRIPT_ONLY
        else:
            mode = SegmentMode.SLIDE
        segments.append(
            Segment(
                segment_id=f"{work.video_id}#{i:03d}",
                video_id=work.video_id,
                t_start=w.t_start,
                t_end=w.t_end,
                mode=mode,
                candidate_ref=w.slide_id,
                slide_ref=w.slide_id,  # 暫定；S4 判定不是投影片時會清掉
                cue_indices=cue_indices,
                transcript_raw="".join(c.text_raw for c in picked),
                transcript_corrected="".join(c.text_corrected or c.text_raw for c in picked),
                corrections=[c for cue in picked for c in cue.corrections],
                boundary_method=(method if i > 0 else BoundaryMethod.VIDEO_BOUNDS),
            )
        )

    work.segments.write_text(
        "[" + ",".join(s.model_dump_json() for s in segments) + "]", encoding="utf-8"
    )
    log.info("S3 %s：%d 個 segment（%s）", work.video_id, len(segments), method.value)
    return segments
