"""S0–S3：本地管線，不花額度。SDD §4.1–§4.6，實作屬 Phase 1（§7.2）。

每個函式的簽名即為 SDD §4 的介面契約。Phase 1 只需把 `pending(...)` 換成
實作，不應改動簽名——簽名改了代表契約改了，那要先改 SDD。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

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


def s1a_transcript(cfg: Config, work: WorkPaths, lexicon: Lexicon | None = None) -> Transcript:
    """S1a 逐字稿。SDD §4.2。

    冪等鍵：video_id + whisper_model + params_hash
    失敗行為：Whisper OOM → 降 batch size 重試一次 → 仍失敗標記 failed 並繼續
    """
    from ..ir import Transcript, TranscriptCue, TranscriptSource, VideoMeta
    from .transcribe import lexicon_prompt, parse_vtt, whisper_transcribe

    meta = VideoMeta.model_validate_json(work.meta.read_text(encoding="utf-8"))
    p = cfg.s1a

    def to_cues(rows: list[tuple[float, float, str]]) -> list[TranscriptCue]:
        return [
            TranscriptCue(index=i, t_start=a, t_end=max(b, a + 0.01), text_raw=t, text_corrected=t)
            for i, (a, b, t) in enumerate(rows)
        ]

    alt_cues = None
    if meta.has_manual_caption and work.captions.exists():
        # 策略 1：手動字幕是人打的，品質遠高於任何 ASR
        rows = parse_vtt(work.captions)
        source, model = TranscriptSource.MANUAL_CAPTION, None
        log.info("S1a %s：採用手動字幕，%d 句", work.video_id, len(rows))
    else:
        # 策略 2／3：跑 Whisper；若另有自動字幕則一併保留供交叉檢查
        prompt = lexicon_prompt(lexicon, p.lexicon_prompt_max_terms) if p.use_lexicon_prompt else None
        rows = whisper_transcribe(work.video, cfg, prompt)
        source, model = TranscriptSource.WHISPER, p.whisper_model
        if meta.has_auto_caption and work.captions.exists():
            alt_cues = to_cues(parse_vtt(work.captions))
        log.info("S1a %s：Whisper %s，%d 句", work.video_id, p.whisper_model, len(rows))

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
    )
    work.transcript.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return transcript


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
    import json

    from .ocr import run_ocr

    if not slides:
        work.ocr.write_text("[]", encoding="utf-8")
        return slides

    images = [work.dir / s.image_path for s in slides]
    results = run_ocr(images, cfg.s2)

    for slide, (text, conf) in zip(slides, results):
        slide.ocr_text = text
        slide.ocr_confidence = min(1.0, max(0.0, conf))

    work.ocr.write_text(
        json.dumps(
            [
                {"slide_id": s.slide_id, "ocr_text": s.ocr_text, "ocr_confidence": s.ocr_confidence}
                for s in slides
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    empty = sum(1 for s in slides if not (s.ocr_text or "").strip())
    log.info("S2 %s：OCR %d 張，%d 張無文字", work.video_id, len(slides), empty)
    return slides


def s2b_lexicon(cfg: Config, work: WorkPaths, slides: list[Slide], series_id: str | None) -> Lexicon:
    """S2b 術語詞庫。SDD §4.4。scope 為**系列級**，不是單片級。"""
    from ..ir import Lexicon
    from .lexicon import build_lexicon

    existing = None
    if work.lexicon.exists():
        existing = Lexicon.model_validate_json(work.lexicon.read_text(encoding="utf-8"))

    lex = build_lexicon(slides, series_id, cfg.s2b, existing)
    work.lexicon.write_text(lex.model_dump_json(indent=2), encoding="utf-8")
    log.info("S2b %s：詞庫 %d 條（scope=%s）", work.video_id, len(lex.entries), series_id or "單片")
    return lex


def s2c_correct(cfg: Config, work: WorkPaths, transcript: Transcript, lexicon: Lexicon) -> Transcript:
    """S2c 逐字稿術語校正。SDD §4.5。

    **這是本專案品質的關鍵路徑，不是可選功能。**

    失敗行為：詞庫為空 → 跳過，transcript_corrected = transcript_raw
    """
    from ..ir import CandidateSet, Slide
    from .lexicon import correct_transcript

    # 「時間上鄰近的投影片」需要每張投影片的時間區間（§4.5 約束 2）
    windows: list[tuple[float, float, str]] = []
    if work.candidates.exists():
        cand = CandidateSet.model_validate_json(work.candidates.read_text(encoding="utf-8"))
        windows = [(c.t_start, c.t_end, f"slide_{c.index + 1:03d}") for c in cand.candidates]

    before = transcript.raw_hash
    transcript, _ = correct_transcript(transcript, lexicon, windows, cfg.s2c)

    # §5.3 不變量 9：transcript_raw 永不被覆寫。這裡就地驗一次，
    # 而不是等到全片跑完才發現——錯了要能立刻定位到是哪個階段動的。
    if Transcript.compute_raw_hash(transcript.cues) != before:
        raise RuntimeError("S2c 改動了 transcript_raw，違反 SDD §4.5 約束 3")

    work.transcript.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return transcript


def s3_align(
    cfg: Config, work: WorkPaths, transcript: Transcript, candidates: CandidateSet
) -> list[Segment]:
    """S3 對齊。SDD §4.6。**這一步不呼叫 LLM**，避免與 S4 形成循環依賴。

    冪等鍵：transcript_hash + candidates_hash + embedding_model
    """
    from ..ir import BoundaryMethod, Segment, SegmentMode, Slide
    from .align import Encoder, assign_cues, coarse_windows, snap_boundary

    p = cfg.s3
    duration = candidates.duration
    windows = coarse_windows(candidates.candidates, duration, p.min_segment_sec)

    ocr_by_slide: dict[str, str] = {}
    if work.ocr.exists():
        import json

        for row in json.loads(work.ocr.read_text(encoding="utf-8")):
            ocr_by_slide[row["slide_id"]] = row.get("ocr_text") or ""

    # 邊界吸附需要 embedding，而 embedding 需要投影片文字。沒有 OCR 就
    # 只能停在粗切——這是降級，不是失敗（§4.3 的 transcript_only 同理）。
    shifts: list[float] = [0.0] * len(windows)
    method = BoundaryMethod.SLIDE_SWITCH
    if ocr_by_slide and len(windows) > 1:
        encoder = Encoder(p.embedding_model, p.device)
        method = BoundaryMethod.SEMANTIC_SNAP
        for i in range(1, len(windows)):
            prev_text = ocr_by_slide.get(windows[i - 1].slide_id or "", "")
            next_text = ocr_by_slide.get(windows[i].slide_id or "", "")
            snapped, shift = snap_boundary(
                windows[i].t_start, transcript.cues, prev_text, next_text,
                encoder, p.snap_window_sec,
            )
            if snapped <= windows[i - 1].t_start or snapped >= windows[i].t_end:
                continue  # 吸附後會讓區間倒轉，放棄這次吸附
            windows[i - 1] = type(windows[i - 1])(windows[i - 1].t_start, snapped, windows[i - 1].slide_id)
            windows[i] = type(windows[i])(snapped, windows[i].t_end, windows[i].slide_id)
            shifts[i] = shift

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
                slide_ref=w.slide_id,
                cue_indices=cue_indices,
                transcript_raw="".join(c.text_raw for c in picked),
                transcript_corrected="".join(c.text_corrected or c.text_raw for c in picked),
                corrections=[c for cue in picked for c in cue.corrections],
                boundary_method=method if i > 0 else BoundaryMethod.VIDEO_BOUNDS,
                boundary_shift_sec=shifts[i],
            )
        )

    work.segments.write_text(
        "[" + ",".join(s.model_dump_json() for s in segments) + "]", encoding="utf-8"
    )
    log.info("S3 %s：%d 個 segment（吸附中位位移 %.1fs）",
             work.video_id, len(segments),
             float(np.median([abs(x) for x in shifts])) if shifts else 0.0)
    return segments
