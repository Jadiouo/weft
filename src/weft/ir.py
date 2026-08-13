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
    SLIDE_SWITCH = "slide_switch"  # 純時間戳粗切，未吸附（v0.4 以前的主幹）
    SEMANTIC_SNAP = "semantic_snap"  # embedding 吸附後
    #: v0.5 的主幹：逐字稿的話題邊界（TextTiling）。**不看畫面。**
    TOPIC_SHIFT = "topic_shift"
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


class CorrectionMethod(StrEnum):
    """術語校正的來源。

    v0.3 移除了本地 OCR + 詞庫鏈（原 `lexicon`），改由 S4 的 VLM 在
    同一次呼叫中對照投影片畫面修正逐字稿。

    **v0.4 讓 `lexicon` 回來了，但來源不同**：詞庫改由 `slide_text`
    （VLM 讀出的繁體）建立，不再依賴本地 OCR。且它只**提案**，
    仍走 R13 的三道閘。見 docs/decisions.md D25。
    """

    VLM = "vlm"
    LEXICON = "lexicon"


class Correction(Strict):
    """SDD §3.3 corrections[]。每次替換都必須留下這筆紀錄。

    `reason` 是 VLM 給的理由，供人工稽核——沒有它就只剩「模型說要改」，
    §5.6 的抽檢無從判斷改得對不對。
    """

    from_text: str = Field(alias="from")
    to_text: str = Field(alias="to")
    source: str  # slide_id（對照投影片改的）或 segment_id
    method: CorrectionMethod = CorrectionMethod.VLM
    reason: str = ""
    #: VLM 不給數值分數，故為選填。保留欄位是為了讓未來可能的
    #: 二次驗證（例如拼音相似度複核）有地方寫。
    score: Score | None = None

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
    #: 若 ASR 輸出經過簡繁轉換，記下用的 OpenCC 設定（例如 `"s2twp"`）。
    #: `text_raw` 存的是**轉換後**的文字——轉換發生在建立 Transcript 之前，
    #: 所以不違反「`text_raw` 永不覆寫」（§5.3 不變量 9）。
    #: 記在這裡是為了讓「這份逐字稿被動過字集」這件事可稽核（D24）。
    script_conversion: str | None = None

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
    """單一抽出幀。02_candidates.json 的原料。

    v0.3 移除了 speaker/slide 二分類——CV 只負責找靜止區段，分類交給 VLM。
    `frame_class` 保留為 `slide`（意為「候選幀」），欄位留著是為了讓
    02_candidates.json 的既有讀取端不必改。
    """

    t: Seconds
    frame_class: FrameClass = FrameClass.SLIDE
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
    """一個靜止區段的代表幀。SDD §3.2。

    v0.3 起這是**候選幀**——S1b 只負責找出畫面靜止的區段並取代表幀，
    「這張圖是不是投影片」由 S4 的 VLM 判定（`Understanding.is_slide`）。
    """

    slide_id: str
    image_path: str
    t_first_seen: Seconds
    t_last_seen: Seconds
    is_progressive_final: bool = False
    build_frames: list[Seconds] = Field(default_factory=list)
    #: 這張投影片出現過的**所有**時間區間（S1c 去重後，§4.3b）。
    #: 未跑去重時為單一區間 `[(t_first_seen, t_last_seen)]`——
    #: 下游可以無條件依賴它，不必判斷有沒有去重過。
    occurrences: list[tuple[Seconds, Seconds]] = Field(default_factory=list)
    #: 被判為同一張時，指向代表幀的 `slide_id`；代表幀自己為 `None`。
    #: **被合併的候選幀不刪除**——§5.6 的人工抽檢要能複核
    #: 「這兩張真的是同一張嗎」。
    duplicate_of: str | None = None
    #: 投影片上的文字，由 **S4 的 VLM** 讀出（v0.3 之前是本地 OCR）。
    #: 這同時是 §5.4 溯源檢查對 slide_ocr 型 block 的比對來源——
    #: 因此 prompt 必須要求 VLM **先逐字轉錄、再詮釋**，見 known-risks R9。
    slide_text: str | None = None
    layout_description: str | None = None  # 由 S4a 填入
    #: S4a 判定「這不是投影片」時的理由（§4.7a）。
    #: 判斷是 S4a 做的，理由就存在這裡——§5.6 的人工複核靠它。
    reject_reason: str | None = None
    #: 第二個模型的描述與本張 `layout_description` 的一致度（R23）。
    #: `None` 表示沒跑這道檢查。**低分不阻擋產出**，只是把 §5.6 的
    #: 人工抽檢引導過來——`圖表描述` 的溯源分離度是 0.00x，
    #: 本來就沒有自動閘門可用。
    description_agreement: Score | None = None


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
    #: S1b 取出的候選幀。**一定會設**（只要該時段有候選），與 VLM 的判定無關。
    #: 保留它是為了讓 debug markdown 能顯示「VLM 判定不是投影片的那張圖」，
    #: 否則被拒絕的候選就消失了，§5.6 的人工抽檢無從複核。
    candidate_ref: str | None = None
    #: 僅在 S4 判定 `is_slide=true` 後才設。
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
    #: 未通過的**成因分類**，由 §5.4 填入。三種未通過的修法完全不同
    #: （R27、票 01），混在一個 `unverified` 裡等於看不見：
    #:   `wrong_source`            內容溯得到同一段的**另一個**來源 → 修 S4c 的 prompt
    #:   `depends_on_correction`   只靠 S4b 的術語校正才對得上 → 實質溯到投影片
    #:   兩者皆否                   兩個來源都對不上 → 才是真的要判內容
    wrong_source: bool = False
    depends_on_correction: bool = False


class Understanding(Strict):
    """07_understanding/seg_NNN.json。SDD §3.4。"""

    #: **VLM 的分類判定**（v0.3）：這個候選幀是不是投影片？
    #: false → 該 segment 降級為 speaker_only，slide_ref 清空。
    is_slide: bool = True
    reject_reason: str | None = None  # is_slide=false 時的理由，供稽核
    #: VLM 對照投影片畫面讀出的文字。§5.4 對 slide_ocr 型 block 的比對來源。
    slide_text: str | None = None
    #: 對照投影片修正的逐字稿錯字。每筆都帶理由，供 §5.6 抽檢。
    corrections: list[Correction] = Field(default_factory=list)
    summary: str
    layout_description: str | None = None
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    model_used: str | None = None  # §5.5 #6：本地 fallback 必須留痕
    prompt_version: str | None = None
    #: 產生這份理解時，那個 segment 的**輸入指紋**（逐字稿 + 投影片）。
    #:
    #: §4.7 的冪等鍵原本只有 `segment_id + prompt_version + model`，而
    #: `segment_id` 是**位置編號**（`#010`）。v0.5 換掉分段方式之後，
    #: `#010` 涵蓋的時間範圍完全不同，**舊快取照樣命中**——實測
    #: cxrqHABhWOU 的 `#010` 現在是 564–593 秒，快取裡卻是 72–98 秒的內容。
    #: 溯源通過率因此從 0.929 掉到 0.071，而**沒有任何機械檢查會抓到**：
    #: 每個欄位都在、格式都對，只是講的是別的地方的事。這與 D20 的
    #: 圖片錯位同一類。
    input_fingerprint: str | None = None
    #: 產生這份結果時的取樣溫度。**留著是為了事後查得出來**
    #: 「這批資料是哪個溫度下產的」——冪等鍵看的是下面的指紋。
    temperature: float | None = None
    #: **所有取樣參數的指紋**（temperature + seed + top_k）。
    #:
    #: 為什麼不逐個欄位比對：這個 repo 已經六次因為「冪等鍵沒涵蓋某個
    #: 決定結果的東西」而量出假結論（D20／D22／D30／D32／α／溫度）。
    #: 每加一個取樣參數就要記得改比對邏輯，**那個「記得」遲早會失效**。
    #:
    #: 改成單一指紋之後，加參數只要改 `sampling_fingerprint()` 一個地方，
    #: 而那個函式有測試釘住「新參數必須讓指紋改變」。
    #:
    #: 舊快取沒有這欄（None）→ 保守重跑。
    sampling_fingerprint: str | None = None


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
    #: `text` 的 sha256 前 16 碼。**下游用來偵測「id 沒變但內容變了」。**
    #:
    #: `Chunk.id` 是 `<video>#<段序號>#b<塊序號>`——**位置編號**。
    #: 改 `block_chars`、換分段方法、甚至換一版 S4c prompt，
    #: `#010` 都還是 `#010`，但它指的時間範圍與內容會整個換掉。
    #:
    #: 這正是 D32：當時 segment_id 的位置性讓 S4c 讀到別的時間範圍的快取，
    #: 而**所有機械檢查都是綠的**。D32 修的是快取鍵；**匯出的 id 沒修**，
    #: 於是同一個陷阱會原封不動地跟著 chunk 進到 vault ——
    #: 以 id 當筆記識別的話，重跑一次就會在同一則筆記下換掉內容。
    #:
    #: 這裡不改 id 的形狀（那要下游先決定用什麼當識別），
    #: 只保證**內容變了看得出來**。見票 13 的稽核。
    content_sha: str
    #: **這支影片的溯源通過率**（D34，2026-08-13）。
    #:
    #: 這個 chunk 自己一定是通過溯源的——未通過的 block 在 S6 就被排除了。
    #: 這一欄講的是**它的出身**：同一支影片裡有多少比例的內容溯不到。
    #:
    #: 為什麼需要它：per-video 閘門原本會把通過率 < 0.95 的影片整支丟掉，
    #: 而實測 8 支素材時**那樣丟掉的已驗證內容（218 block）比留下的（135）
    #: 還多**。D34 改為不丟，但下游要有依據判斷——低通過率代表檢查器在
    #: 那批素材上吃力（R42：STEM 上具名實體檢查空轉、相似度門檻只有 0.05），
    #: 所以「通過」在那裡是較弱的證據。
    #:
    #: **給 vault 側的建議**：先全收，等實際查過之後再決定要不要按這一欄過濾。
    #:
    #: **可為 `None`**：溯源還沒跑時就是不知道。把它填成 1.0 會讓
    #: 「沒檢查過」看起來像「完美通過」——那比缺值更糟。
    #: 生產路徑（`s6_render`）一定會先跑 `check_video`，所以那裡永遠有值；
    #: `test_unit_render.py` 有一條釘住這件事。
    video_pass_rate: float | None


class Chunk(Strict):
    """chunks.jsonl 每一行。"""

    id: str
    text: str
    metadata: ChunkMetadata


Segment.model_rebuild()
