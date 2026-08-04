"""設定檔。每個階段的設定物件自己會算 params_hash（SDD §6.3）。

注意：§5.2 的量化門檻**不在這裡**。門檻寫死在 `weft.validation.thresholds`，
不可經設定檔覆寫——那正是 §5.5 #7 禁止的事。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .state import params_hash


class StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def params_hash(self) -> str:
        return params_hash(self.model_dump(mode="json"))


class S0Config(StageConfig):
    """取得。SDD §4.1。"""

    video_format: str = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
    caption_langs: list[str] = Field(default_factory=lambda: ["zh-TW", "zh-Hant", "zh", "zh-CN"])
    prefer_manual_caption: bool = True
    write_auto_caption: bool = True
    rate_limit: str | None = None


class S1aConfig(StageConfig):
    """逐字稿。SDD §4.2。"""

    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "zh"
    beam_size: int = 5
    vad_filter: bool = True
    #: 把系列術語詞庫餵進 initial_prompt（§9 文言文 ASR 的緩解）
    use_lexicon_prompt: bool = True
    lexicon_prompt_max_terms: int = 40


class S1bConfig(StageConfig):
    """投影片候選幀。SDD §4.3。"""

    fps: float = 1.0
    #: 降解析度短邊，壓制雷射筆與壓縮雜訊（§4.3 步驟 3）
    downscale_short_side: int = 180
    blur_sigma: float = 2.0
    #: speaker/slide 二分類
    face_detector: str = "yunet"
    face_min_area_ratio: float = 0.04  # 滿版人臉的下限面積佔比
    #: HMM 換頁偵測（§4.3 步驟 4）——以「投影片會停留一段時間」為先驗。
    #: 0.97 對應幾何分布平均停留 ~33 幀，在 1fps 下即 ~33 秒，與 §5.1 的
    #: 「每頁停留 30–120 秒」相符。
    hmm_self_transition: float = 0.97
    #: 「同一頁」基線離散度的下限，作用在**尺度無關**的 ink Jaccard 上：
    #: 語意為「換頁至少會改變這個比例的 ink 圖樣」。沒有它，完全沒換頁的
    #: 影片（對抗樣本 A4）會把量化雜訊當成離群值。見 docs/decisions.md D7。
    min_ink_change: float = 0.01
    min_slide_duration_sec: float = 3.0
    #: 逐條動畫合併（§4.3 步驟 5）。0.70 取自實測分離區間：
    #: build 的 containment 0.86–0.98，真正換頁 ≤0.56。見 decisions.md D8。
    progressive_merge: bool = True
    progressive_containment_ratio: float = 0.70


class S2Config(StageConfig):
    """OCR。SDD §4.4。"""

    ocr_model: str = "PaddleOCR-VL"
    lang: str = "ch"
    use_gpu: bool = False
    #: PaddleOCR 的 `ch` 模型輸出簡體，但本專案素材是繁體。不轉的話詞庫會
    #: 混入簡體詞條，而 §4.5 是拿詞庫比對繁體逐字稿——永遠匹配不到。
    normalise_to_traditional: bool = True


class S2bConfig(StageConfig):
    """術語詞庫萃取。SDD §4.4。"""

    term_min_len: int = 2
    term_max_len: int = 6
    min_count: int = 1
    #: 書名號、括號內文字視為術語候選
    bracket_extraction: bool = True


class S2cConfig(StageConfig):
    """逐字稿術語校正。SDD §4.5。"""

    similarity_threshold: float = 0.85
    #: 只在時間上鄰近的投影片詞庫中比對（±N 個候選段）
    neighbor_window: int = 2
    pinyin_weight: float = 0.7
    shape_weight: float = 0.3


class S3Config(StageConfig):
    """對齊。SDD §4.6。"""

    embedding_model: str = "BAAI/bge-m3"
    device: str = "cuda"
    #: 吸附範圍硬限制（§4.6 關鍵約束）。放大此值等同繞過設計約束。
    snap_window_sec: float = 20.0
    min_segment_sec: float = 5.0


class S4Config(StageConfig):
    """聯合理解。唯一花額度的階段。SDD §4.7。"""

    model: str = "gemini-2.5-flash-lite"
    prompt_version: str = "v1"
    #: 可將 2–3 個相鄰 segment 併為一次呼叫，但輸出仍逐 segment 分開
    batch_segments: int = 3
    prev_summary_max_chars: int = 200
    max_retries: int = 2
    retry_backoff_sec: float = 4.0
    #: §5.5 #6：本地 fallback 必須是明確開關，且輸出須標記 model_used。
    #: 預設關閉——額度耗盡時停止，不靜默降級。
    allow_local_fallback: bool = False
    local_fallback_model: str = "Qwen/Qwen3-VL-8B-Instruct"


class S5Config(StageConfig):
    model: str = "gemini-2.5-flash-lite"
    prompt_version: str = "v1"


class S6Config(StageConfig):
    """渲染。SDD §4.9。"""

    max_chunk_chars: int = 800
    write_debug_markdown: bool = True


class QuotaConfig(StageConfig):
    """SDD §6.1 / §6.2。"""

    #: 主動節流水位。超過即停止本日處理，不靠撞 429（§5.5 #13）。
    safety_ratio: float = 0.9
    requests_per_day: int = 1000
    #: RPD 於太平洋時間午夜重置——用 zoneinfo 算，不寫死時差（§6.2）
    reset_timezone: str = "America/Los_Angeles"


class ProvenanceConfig(StageConfig):
    """SDD §5.4 溯源檢查。門檻分 content_type 設定。

    這裡的數值是「檢查怎麼跑」的參數，不是 §5.2 的驗收門檻——
    驗收門檻（溯源通過率 ≥ 0.95）在 thresholds.py，不可調。
    """

    #: 正向檢查門檻，對 bigram containment。0.25 取自實測分離區間的幾何中點：
    #: 忠實改寫落在 ≥0.364，重用術語的對抗性幻覺落在 ≤0.154（見 docs/decisions.md）。
    #: 注意「編造書名／編造數字」這類插入單一假細節的幻覺相似度本來就高，
    #: 靠的是具名實體檢查而非這個門檻——調高它擋不到那類，只會誤殺改寫。
    min_similarity: float = 0.25
    #: 反向檢查：逐字複製率過高代表 LLM 只在複製貼上
    copy_similarity: float = 0.98
    #: 各型別允許的高複製率 block 比例上限。經文原文本來就該是引文。
    max_copy_ratio: dict[str, float] = Field(
        default_factory=lambda: {
            "經文原文": 1.0,
            "圖表描述": 0.5,
            "白話解說": 0.8,
            "口頭延伸": 0.8,
        }
    )
    check_named_entities: bool = True


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_dir: Path = Path("work")
    out_dir: Path = Path("out")
    log_level: str = "INFO"

    s0: S0Config = Field(default_factory=S0Config)
    s1a: S1aConfig = Field(default_factory=S1aConfig)
    s1b: S1bConfig = Field(default_factory=S1bConfig)
    s2: S2Config = Field(default_factory=S2Config)
    s2b: S2bConfig = Field(default_factory=S2bConfig)
    s2c: S2cConfig = Field(default_factory=S2cConfig)
    s3: S3Config = Field(default_factory=S3Config)
    s4: S4Config = Field(default_factory=S4Config)
    s5: S5Config = Field(default_factory=S5Config)
    s6: S6Config = Field(default_factory=S6Config)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)
    provenance: ProvenanceConfig = Field(default_factory=ProvenanceConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        if path is None:
            return cls()
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)
