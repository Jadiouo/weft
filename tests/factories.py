"""建構「一份完全合法的 IR」，供不變量測試當作正例基準。

反例測試的作法一律是：從這裡拿一份合法資料，只破壞**一個**性質，然後斷言
對應的那一條規則（且只有那一條）抓到它。這樣才能證明檢查器抓的是它宣稱要
抓的東西，而不是碰巧一起紅燈。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from weft.ir import (
    BoundaryMethod,
    ContentBlock,
    ContentType,
    Correction,
    Provenance,
    ProvenanceKind,
    Segment,
    SegmentMode,
    Slide,
    Transcript,
    TranscriptCue,
    TranscriptSource,
    Understanding,
    VideoIR,
    VideoMeta,
)

VIDEO_ID = "zIglvjoU9vo"
SERIES_ID = "PL37L2Y8jeuGDeSRi5ntD_gK1pBM8KlSkv"

SLIDE_OCR = "一月為胞，精血凝也。二月為胎，形兆胚也。三月陽神為三魂，動而生也。"


def make_transcript() -> Transcript:
    raw = [
        (0.0, 20.0, "今天我們講太上老君內觀經的第一段。"),
        (20.0, 40.0, "經文說一月為胞，精血凝也。"),
        (40.0, 60.0, "這裡的時運其實是識蘊，講的是入胎的時機。"),
        (60.0, 90.0, "講者用簽約來比喻這個過程。"),
    ]
    cues = [
        TranscriptCue(index=i, t_start=a, t_end=b, text_raw=t, text_corrected=t)
        for i, (a, b, t) in enumerate(raw)
    ]
    # 第 2 句有一處術語校正：時運 → 識蘊
    cues[2].text_corrected = cues[2].text_raw.replace("時運", "識蘊")
    cues[2].corrections = [
        Correction(**{"from": "時運", "to": "識蘊", "source": "slide_001",
                      "method": "vlm", "reason": "投影片作「識蘊」，逐字稿同音誤植"})
    ]
    return Transcript(
        video_id=VIDEO_ID,
        source=TranscriptSource.MANUAL_CAPTION,
        cues=cues,
        raw_hash=Transcript.compute_raw_hash(cues),
    )


def make_ir(base_dir: Path, with_understanding: bool = True) -> VideoIR:
    """完整合法的 VideoIR。會實際在 base_dir 寫出投影片圖（不變量 5 要求可開啟）。"""
    slides_dir = base_dir / "03_slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 180), (250, 248, 243)).save(slides_dir / "slide_001.png")

    slides = [
        Slide(
            slide_id="slide_001",
            image_path="03_slides/slide_001.png",
            t_first_seen=20.0,
            t_last_seen=60.0,
            is_progressive_final=True,
            build_frames=[22.0, 35.0],
            slide_text=SLIDE_OCR,
        )
    ]

    transcript = make_transcript()
    understanding = (
        Understanding(
            summary="本段解說《太上老君內觀經》一月為胞之義。",
            layout_description="上方為標題，下方紫底區塊為經文引文與白話解說。",
            content_blocks=[
                ContentBlock(
                    type=ContentType.SCRIPTURE,
                    text="一月為胞，精血凝也。",
                    provenance=Provenance(kind=ProvenanceKind.SLIDE_OCR, ref="slide_001"),
                ),
                ContentBlock(
                    type=ContentType.ORAL,
                    text="講者以簽約作比喻，說明識蘊進入的時機。",
                    provenance=Provenance(kind=ProvenanceKind.TRANSCRIPT, ref="40.0-90.0"),
                ),
            ],
            terms=["識蘊", "胞", "精血"],
            model_used="gemini-2.5-flash-lite",
        )
        if with_understanding
        else None
    )

    segments = [
        Segment(
            segment_id=f"{VIDEO_ID}#000",
            video_id=VIDEO_ID,
            t_start=0.0,
            t_end=20.0,
            mode=SegmentMode.SPEAKER_ONLY,
            cue_indices=[0],
            transcript_raw=transcript.cues[0].text_raw,
            transcript_corrected=transcript.cues[0].text_raw,
            boundary_method=BoundaryMethod.SLIDE_SWITCH,
        ),
        Segment(
            segment_id=f"{VIDEO_ID}#001",
            video_id=VIDEO_ID,
            t_start=20.0,
            t_end=90.0,
            mode=SegmentMode.SLIDE,
            candidate_ref="slide_001",
            slide_ref="slide_001",
            cue_indices=[1, 2, 3],
            transcript_raw="".join(c.text_raw for c in transcript.cues[1:]),
            transcript_corrected="".join(c.text_corrected or c.text_raw for c in transcript.cues[1:]),
            corrections=list(transcript.cues[2].corrections),
            boundary_method=BoundaryMethod.SEMANTIC_SNAP,
            boundary_shift_sec=-6.2,
            understanding=understanding,
        ),
    ]

    return VideoIR(
        meta=VideoMeta(
            video_id=VIDEO_ID,
            title="太上老君內觀經講記（三）",
            duration=90.0,
            url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
            series_id=SERIES_ID,
            episode_index=3,
            has_manual_caption=True,
        ),
        slides=slides,
        segments=segments,
    )
