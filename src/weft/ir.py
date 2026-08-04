"""中介表示（IR）—— 整個系統的契約。SDD §3。

所有階段圍繞這些型別。`extra="forbid"` 是刻意的：階段之間偷加欄位會讓
契約悄悄漂移，寧可在反序列化時就炸掉。
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# 封閉列舉
# --------------------------------------------------------------------------


class ContentType(StrEnum):
    """SDD §3.4：content_blocks[].type 的封閉列舉。"""

    SCRIPTURE = "經文原文"
    VERNACULAR = "白話解說"
    FIGURE = "圖表描述"
    ORAL = "口頭延伸"


class ProvenanceKind(StrEnum):
    SLIDE_OCR = "slide_ocr"
    TRANSCRIPT = "transcript"


class SegmentMode(StrEnum):
    """SDD §3.3。transcript_only 為全片無投影片的降級模式（§4.3 失敗行為）。"""

    SLIDE = "slide"
    SPEAKER_ONLY = "speaker_only"
    TRANSCRIPT_ONLY = "transcript_only"


class TranscriptSource(StrEnum):
    """SDD §4.2 的字幕優先序。"""

    MANUAL_CAPTION = "manual_caption"
    WHISPER = "whisper"
    AUTO_CAPTION = "auto_caption"


class BoundaryMethod(StrEnum):
    SLIDE_SWITCH = "slide_switch"  # 純時間戳粗切，未吸附
    SEMANTIC_SNAP = "semantic_snap"  # embedding 吸附後
    VIDEO_BOUNDS = "video_bounds"  # transcript_only：整片一段


class FrameClass(StrEnum):
    SPEAKER = "speaker"
    SLIDE = "slide"


class VerificationStatus(StrEnum):
    """SDD §5.4 溯源檢查結果。"""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    DEGENERATE_COPY = "degenerate_copy"


Seconds = Annotated[float, Field(ge=0.0)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


# --------------------------------------------------------------------------
# S0 — 取得
# --------------------------------------------------------------------------


class VideoMeta(Strict):
    """00_meta.json。SDD §4.1。"""

    video_id: str
    title: str
    duration: Seconds
    url: str
    series_id: str | None = None
    episode_index: int | None = None
    upload_date: str | None = None
    has_manual_caption: bool = False
    has_auto_caption: bool = False
    caption_lang: str | None = None
    video_path: str | None = None
    caption_path: str | None = None


# --------------------------------------------------------------------------
# S1a / S2c — 逐字稿與術語校正
# --------------------------------------------------------------------------


class Correction(Strict):
    """SDD §3.3 corrections[]。每次替換都必須留下這筆紀錄。"""

    from_text: str = Field(alias="from")
    to_text: str = Field(alias="to")
    source: str  # slide_id
    method: Literal["lexicon"]
    score: Score

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TranscriptCue(Strict):
    """一句逐字稿。text_raw 一旦寫入即為唯讀（§5.3 不變量 9）。"""

    index: int = Field(ge=0)
    t_start: Seconds
    t_end: Seconds
    text_raw: str
    text_corrected: str | None = None
    corrections: list[Correction] = Field(default_factory=list)


class Transcript(Strict):
    """05_transcript.json。S1a 建立，S2c 就地更新。"""

    video_id: str
    source: TranscriptSource
    language: str = "zh"
    cues: list[TranscriptCue]
    raw_hash: str  # §5.3 #9 的比對基準
    alt_cues: list[TranscriptCue] | None = None  # §4.2 策略 3 的交叉檢查用
    model: str | None = None
    params_hash: str | None = None

    @staticmethod
    def compute_raw_hash(cues: list[TranscriptCue]) -> str:
        """對 (index, t_start, t_end, text_raw) 取 hash。時間戳也納入，
        因為「未被修改」包含時間軸未被偷偷平移。"""
        h = hashlib.sha256()
        for cue in cues:
            h.update(f"{cue.index}|{cue.t_start:.3f}|{cue.t_end:.3f}|{cue.text_raw}\n".encode())
        return h.hexdigest()

    def raw_is_intact(self) -> bool:
        return self.compute_raw_hash(self.cues) == self.raw_hash


# --------------------------------------------------------------------------
# S1b — 投影片候選幀
# --------------------------------------------------------------------------


class FrameLabel(Strict):
    """單一抽出幀的分類結果。02_candidates.json 的原料。"""

    t: Seconds
    frame_class: FrameClass
    face_score: Score | None = None
    frame_path: str | None = None


class SlideCandidate(Strict):
    """HMM 換頁偵測輸出的一個穩定段落。"""

    index: int = Field(ge=0)
    t_start: Seconds
    t_end: Seconds
    keyframe_t: Seconds  # 取作代表的那一幀（逐條動畫時為最後一幀）
    build_frames: list[Seconds] = Field(default_factory=list)
    merged_from: list[int] = Field(default_factory=list)  # 被合併掉的候選段 index


class CandidateSet(Strict):
    """02_candidates.json。"""

    video_id: str
    fps: float = Field(gt=0)
    duration: Seconds
    frames: list[FrameLabel]
    candidates: list[SlideCandidate]
    params_hash: str


# --------------------------------------------------------------------------
# S1b / S2 — Slide
# --------------------------------------------------------------------------


class Slide(Strict):
    """SDD §3.2。"""

    slide_id: str
    image_path: str
    t_first_seen: Seconds
    t_last_seen: Seconds
    is_progressive_final: bool = False
    build_frames: list[Seconds] = Field(default_factory=list)
    ocr_text: str | None = None
    ocr_confidence: Score | None = None
    layout_description: str | None = None  # 由 S4 填入


# --------------------------------------------------------------------------
# S2b — 術語詞庫（scope 為系列級，SDD §4.4）
# --------------------------------------------------------------------------


class LexiconEntry(Strict):
    term: str
    pinyin: str | None = None
    count: int = Field(default=1, ge=1)
    first_seen: dict[str, list[str]] = Field(default_factory=dict)  # video_id -> [slide_id]


class Lexicon(Strict):
    """04_lexicon.json。可跨影片累積。"""

    series_id: str | None = None
    entries: list[LexiconEntry] = Field(default_factory=list)
    ocr_model: str | None = None


# --------------------------------------------------------------------------
# S3 — Segment
# --------------------------------------------------------------------------


class Segment(Strict):
    """SDD §3.3。"""

    segment_id: str
    video_id: str
    t_start: Seconds
    t_end: Seconds
    mode: SegmentMode
    slide_ref: str | None = None
    cue_indices: list[int] = Field(default_factory=list)  # §5.3 #3 的驗證依據
    transcript_raw: str = ""
    transcript_corrected: str = ""
    corrections: list[Correction] = Field(default_factory=list)
    boundary_method: BoundaryMethod
    boundary_shift_sec: float = 0.0
    understanding: Understanding | None = None


# --------------------------------------------------------------------------
# S4 — Understanding
# --------------------------------------------------------------------------


class Provenance(Strict):
    """SDD §3.4：必填，不得為 null。§5.3 #7。"""

    kind: ProvenanceKind
    ref: str  # slide_id，或 "748.3-845.1" 形式的時間區間


class ContentBlock(Strict):
    type: ContentType
    text: str
    provenance: Provenance  # 無 default —— 缺了就無法建構
    verification: VerificationStatus | None = None  # 由 §5.4 溯源檢查填入
    similarity: Score | None = None


class Understanding(Strict):
    """07_understanding/seg_NNN.json。SDD §3.4。"""

    summary: str
    layout_description: str | None = None
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    model_used: str | None = None  # §5.5 #6：本地 fallback 必須留痕
    prompt_version: str | None = None


# --------------------------------------------------------------------------
# 最終 IR 與輸出
# --------------------------------------------------------------------------


class VideoIR(Strict):
    """08_video.json。"""

    meta: VideoMeta
    slides: list[Slide] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    tldr: str | None = None  # S5
    term_index: list[str] = Field(default_factory=list)  # S5
    chapters: list[dict] = Field(default_factory=list)  # S5
    needs_review: bool = False  # §5.4：unverified 比例超標
    unverified_ratio: float | None = None

    def slide_by_id(self, slide_id: str) -> Slide | None:
        return next((s for s in self.slides if s.slide_id == slide_id), None)


class ChunkMetadata(Strict):
    """SDD §3.5。§5.3 #8 要求全欄位完整、無 null——
    故此處只有 v2 預留欄位允許 None，其餘皆必填。"""

    video_id: str
    series_id: str | None  # v2 預留（§7.5），playlist 來源時必有
    video_title: str
    episode_index: int | None  # 同上
    t_start: Seconds
    t_end: Seconds
    url: str
    content_type: ContentType
    slide_ref: str | None  # speaker_only 段落無投影片
    terms: list[str]
    provenance_kind: ProvenanceKind


class Chunk(Strict):
    """chunks.jsonl 每一行。"""

    id: str
    text: str
    metadata: ChunkMetadata


Segment.model_rebuild()
