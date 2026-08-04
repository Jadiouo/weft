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


class S1bConfig(StageConfig):
    """投影片候選幀。SDD §4.3。"""

    fps: float = 1.0
    #: 降解析度短邊，壓制雷射筆與壓縮雜訊（§4.3 步驟 3）
    downscale_short_side: int = 180
    blur_sigma: float = 2.0
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


class S3Config(StageConfig):
    """對齊。SDD §4.6。

    **v0.3 移除了語意邊界吸附。** 吸附需要投影片文字來判斷「這句話比較像
    前一張還是後一張」，而 v0.3 拿掉了本地 OCR——投影片文字要到 S4 才有，
    S3 拿不到。§4.6 明文禁止 S3 呼叫 LLM（避免與 S4 循環依賴），所以吸附
    無法在此進行。S3 現在只做粗切。見 known-risks R10。
    """

    min_segment_sec: float = 5.0


class S4Config(StageConfig):
    """聯合理解。唯一花額度的階段。SDD §4.7。"""

    #: **釘住具體版本，不用 `gemini-flash-lite-latest` 這類別名。**
    #: §4.7 的冪等鍵是 `segment_id + prompt_version + model`——別名會在
    #: 底下換掉，讓已花額度的結果與新結果混在一起卻看不出來。
    #:
    #: 2026-08-04 實測：`gemini-2.5-flash-lite` 對新使用者已停用（404
    #: "no longer available to new users"），雖然仍出現在 models.list()。
    model: str = "gemini-3.1-flash-lite"
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
    model: str = "gemini-3.1-flash-lite"
    prompt_version: str = "v1"


class S6Config(StageConfig):
    """渲染。SDD §4.9。"""

    max_chunk_chars: int = 800
    write_debug_markdown: bool = True


class QuotaConfig(StageConfig):
    """SDD §6.1 / §6.2。"""

    #: 主動節流水位。超過即停止本日處理，不靠撞 429（§5.5 #13）。
    safety_ratio: float = 0.9
    #: **初始猜測值，不是事實。** ledger 會從 429 回應中學到真實配額並覆蓋它
    #: （SDD §9 的緩解：「Ledger 讀取實際配額而非寫死」）。
    #:
    #: SDD §6.5 原本寫 1000，實測 gemini flash-lite 的 free tier 是 **20**——
    #: 差 50 倍，主動節流因此完全沒觸發，白燒了一整天配額。預設值改為保守
    #: 的 20，寧可提早停也不要撞牆。
    requests_per_day: int = 20
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
    #: **逐 content_type 的正向門檻。** SDD §5.4 對反向檢查已明訂「檢查須依
    #: type 分別設定範圍」，正向檢查原本卻只有單一門檻——這是規格的不對稱。
    #:
    #: R12 校準實測各型別「忠實 vs 幻覺」的分離度：
    #:   經文原文 7.50x  ← 只有這一型 containment 真的有鑑別力
    #:   白話解說 1.14x
    #:   口頭延伸 0.54x  ← 分布**反轉**（忠實最低 0.090 < 幻覺最高 0.167）
    #:   圖表描述 0.00x  ← 跨語言時忠實與幻覺都是 0.000
    #:
    #: 對後三型，**任何門檻不是擋掉忠實內容就是放行幻覺**。所以它們的門檻
    #: 降到只擋「與來源完全無關」的退化情形，防幻覺的責任移交給具名實體
    #: 檢查與 §5.6 的人工抽檢。**這是實質的保護弱化，見 known-risks R12。**
    min_similarity_by_type: dict[str, float] = Field(
        default_factory=lambda: {
            # 唯一能分辨「忠實 vs 部分編造」的型別（實測分離 7.50x）
            "經文原文": 0.60,
            # 以下三型**分不出忠實與部分編造**（分布重疊，見 R12）。
            # 但它們仍分得出「**引錯來源**」——block 宣稱來自投影片、內容卻
            # 只在逐字稿裡時，containment 約 0.06，低於所有忠實樣本
            # （實測最低 0.125 / 0.090）。所以留一個**只擋引錯來源**的低下限。
            #
            # 這兩件事必須分清楚：
            #   擋得到：引錯來源（provenance 指錯地方）
            #   擋不到：部分編造（真引文接上編造，真的那半把分數撐起來）
            "白話解說": 0.10,
            "口頭延伸": 0.05,
            # 圖表描述**無法設任何下限**：跨語言時忠實者本身就是 0.000
            # （英文投影片 → 中文翻譯）。設 0.0 等於停用，這是實測逼出來的。
            "圖表描述": 0.0,
        }
    )
    #: **來源長度 ÷ block 長度**的下限。擋的是「來源太短，撐不起這段內容」。
    #:
    #: 這是 containment 停用後唯一還抓得到東西的正向檢查。實測案例：
    #: 某個 block 描述「收到一份關於此主題的資料並準備解說」，而該 segment
    #: 的逐字稿只有「講完好像就」5 個字——比值 0.11。
    #:
    #: **注意：這不是校準出來的。** 觀測到的失敗案例只有 n=1（比值 0.11），
    #: 次低的忠實樣本是 0.28（標題卡、文言文簡短句的展開）。0.15 取在兩者
    #: 之間偏保守側，只擋極端情形。累積更多真實資料後應重新校準。
    min_source_ratio: float = 0.15
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
