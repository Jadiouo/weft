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
    #: OpenCC 設定檔，用於把 **ASR 輸出**轉成繁體；`None` 關閉。
    #: 實測 Whisper large-v3 對這批繁體素材輸出 9.43% 簡體字，
    #: 使 §5.4 溯源通過率從 97.2% 掉到 91.5%（R18）。
    #: 用 `s2tw` 而不是 `s2t` 或 `s2twp`（D24）：
    #:   - `s2t` 產出大陸標準字形（`爲`×71、`裏`×35、`纔`、`着`、`喫`），
    #:     與台灣素材的 `為`/`裡`/`才`/`著`/`吃` 對不起來
    #:   - `s2twp` 額外做**詞彙**替換，實測把 `局部`→`區域性`、`運`→`執`、
    #:     `序`→`式`——那是改語意不是轉字集
    #:   - `s2tw` 給台灣字形且不動詞彙，正好是要的
    #: **只作用於 ASR**（Whisper 與自動字幕），手動字幕不轉。
    asr_script_conversion: str | None = "s2tw"


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


class S1cConfig(StageConfig):
    """投影片去重。SDD §4.3b（v0.4 新增）。純本地 CV，不花額度。"""

    enabled: bool = True
    #: 與全片中位幀的距離低於此值即視為攝影棚定鏡。
    #: 實測分離 **6.7x**（攝影棚定鏡最高 0.042、真投影片最低 0.281），
    #: 取中間偏保守側。**零漏抓**。
    studio_distance: float = 0.10
    #: ink Jaccard 單連結的合併門檻。實測 0.20–0.40 結果**完全相同**，
    #: 且沒有任何一群混到不同類別——間隙是素材本身的，不是調出來的。
    jaccard_threshold: float = 0.30
    #: 中位幀的取樣間隔。實測 2519 幀取 1/4 與全讀結果一致。
    median_stride: int = 4
    #: 最大群的佔比上限。超過即判定門檻過鬆，**跳過去重**而非自動調參
    #: （§5.5 #4）。0.9 是保護性上限，不是校準值——正常素材的最大群
    #: 實測佔 17/21（0.81，攝影棚定鏡那群，但它在第 1 道就被剔除了）。
    max_group_ratio: float = 0.9


class S3Config(StageConfig):
    """對齊。SDD §4.6。

    **v0.3 移除了語意邊界吸附。** 吸附需要投影片文字來判斷「這句話比較像
    前一張還是後一張」，而 v0.3 拿掉了本地 OCR——投影片文字要到 S4 才有，
    S3 拿不到。§4.6 明文禁止 S3 呼叫 LLM（避免與 S4 循環依賴），所以吸附
    無法在此進行。S3 現在只做粗切。見 known-risks R10。
    """

    min_segment_sec: float = 5.0


class S4aConfig(StageConfig):
    """投影片理解。SDD §4.7a（v0.4 新增）。

    **逐張相異投影片一次呼叫，不批次**——D20 的錯位就是批次造成的。
    """

    #: `供應者:模型名`。**供應者必須明寫**（§5.5 #6：本地 fallback 要是
    #: 明確的設定開關，猜測會讓那條規定失效）。
    #:
    #: 預設用 gemini 以維持既有行為；改成 `ollama:...` 即不花額度。
    #: R21 實測（跨集 16 張人工繕打）：
    #:   is_slide     gemma4:12b 90.5% > qwen3-vl 84.2% > gemma3 71.4% > qwen2.5vl 66.7%
    #:   slide_text   qwen2.5vl 4.9% < gemma3 7.9% < gemma4 37.8%
    #: **兩項的最佳模型不同**，這正是逐子階段可設定的理由。
    model: str = "gemini:gemini-3.1-flash-lite"
    #: **分類另用一個模型**（§2.3 的 S4a-1）。`None` 表示與 `model` 同一個。
    #:
    #: 存在的理由是實測（R21 + 本機端到端）：分類與轉錄的最佳模型不同，
    #: 而且**分類錯的代價比轉錄錯高**——把講者鏡頭判成投影片，
    #: 攝影棚佈景的書法就成了 §5.4 的「合法來源」，後面所有引用它的
    #: block 都會拿假來源去驗證。實跑 qwen2.5vl 單模型時 7 個誤報，
    #: 溯源未通過比例 24.3%。
    classifier_model: str | None = None
    #: v5（2026-08-08）：兩題都要過——**(一) 有沒有成段可讀的講解內容**、
    #: **(二) 那些內容是不是這一集特有的**。
    #:
    #: 逐版實測（調校集 3 支 58 張代表幀、保留集 1 支 22 張，逐張人工標註）：
    #:   v1 「畫面主體為教材內容」                調校 0.818
    #:   v2 排除主講人簡介，但寫「有講者不影響判定」  調校 0.636
    #:   v3 兩步，第一步「看得到講者 → 不是」        調校 1.000、保留 0.909
    #:   v4 只問「是不是這一集特有的」              調校 0.861、保留 0.952
    #:   v5 加上「成段」這個維度                    見 §5.2 的量測
    #:
    #: **v3 的 1.000 是假的**：那三支全是攝影棚定鏡，20 張正例沒有一張
    #: 看得到講者，「有講者 → 不是」在上面 20/20 成立，其實是對素材過擬合。
    #: 保留集是講堂實景，slide_023/024 左半是可讀的肝門靜脈圖、右半是講者，
    #: v3 的錯誤恰好就是這兩張。
    #: **v4 修好了那兩張，卻漏掉片頭**——講者配一個大字關鍵詞加一行字幕，
    #: 模型看不出「這個詞每集都一樣」，因為它一次只看得到一張圖。
    #: v5 因此改用它看得見的維度：**份量**。片頭是一兩個大字，教材是成段的。
    #: 見 docs/proposals/slide-definition.md。
    prompt_version: str = "v5"
    max_retries: int = 2
    retry_backoff_sec: float = 4.0
    #: **第二個描述模型**，用來對 `description` 做一致性檢查（R23）。
    #: `None` 關閉。
    #:
    #: R23 量到多模型一致性對描述編造的分離度 **13.59x**（事實主張平均被
    #: 1.81/3 個模型提到，編造只有 0.13/3）。這一道**不阻擋產出**——
    #: §5.4 對 `圖表描述` 的分離度是 0.00x，本來就沒有自動閘門可用；
    #: 目的是把 §5.6 的人工抽檢**引導到最可能出錯的那幾張**，
    #: 目前抽檢沒有任何優先序，等於全片平均分配注意力。
    description_checker_model: str | None = None
    #: 兩份描述的 bigram containment 低於此值即標記 `needs_review`。
    #: **未經校準**——R23 量的是「主張層級」的一致性，這裡用的是
    #: 粗得多的整段相似度。先記錄分布，累積後再定。
    description_agreement_min: float = 0.25
    #: **跨集重現前濾**（R26）：在別集也一模一樣出現過的畫面，
    #: 不必問模型就知道是節目包裝——片頭字卡、logo、系列名稱卡、
    #: 主講人學經歷、片尾訂閱畫面，共同點正是每集都一樣。
    #:
    #: 實測（四集、80 張代表幀）：是投影片的跨集最小灰階 MAE 最低 7.57，
    #: 不是的最低 0.44、中位 3.61。門檻 6.0 抓到 29/54，**誤殺 0/26**。
    #: 接在分類前面，四集合計 0.900 → 0.925。
    #:
    #: **這一步讓 S4a 依賴「同系列其他集也處理過」**，而 S-1／S1c 是刻意
    #: 逐集獨立的。因此是明確開關，且參考集不足時明說並跳過，不靜默降級。
    cross_episode_filter: bool = True
    #: 跨集最小 MAE 低於此值即判定為節目包裝。6.0 落在實測的 0.44–7.57
    #: 之間，那段區間裡任何值結果相同——**間隙是素材本身的，不是調出來的**。
    cross_episode_mae: float = 6.0


class S4Config(StageConfig):
    """逐段理解（v0.4 起為 S4c）。SDD §4.7c。"""

    #: **釘住具體版本，不用 `gemini-flash-lite-latest` 這類別名。**
    #: §4.7 的冪等鍵是 `segment_id + prompt_version + model`——別名會在
    #: 底下換掉，讓已花額度的結果與新結果混在一起卻看不出來。
    #:
    #: 2026-08-04 實測：`gemini-2.5-flash-lite` 對新使用者已停用（404
    #: "no longer available to new users"），雖然仍出現在 models.list()。
    model: str = "gemini:gemini-3.1-flash-lite"
    #: **v0.4 預設不送圖**：投影片文字已由 S4a 讀出並以文字掛進 prompt。
    #: 再送一次圖等於讓同一個模型既產生來源又產生待驗證的內容（R9），
    #: 而且會把 D20 的錯位風險帶回來。要送必須明確打開。
    send_images: bool = False
    #: v7（2026-08-08）：`provenance_ref` **不再向模型要**（由管線填），
    #: 並把來源歸屬的說明壓成三句。
    #:
    #: v6 是同一件事的失敗版本：我寫了整整一段（含實測數字表格）解釋怎麼標，
    #: 還寫了「provenance_ref 隨便填」。結果 qwen2.5:14b 被帶偏，
    #: **投影片段落產出 0 個 content_block**——167 個 block 掉到 21 個。
    #: 教訓：給 14B 模型的規則要短。長篇解釋對人有用，對它是雜訊。
    #:
    #: v6 的原意：改寫 `provenance_kind` 的說明。R27 拆解四支影片
    #: 27 筆溯源未通過，**15 筆是摘要投影片卻標成 transcript**——系統於是
    #: 拿逐字稿去驗證一段其實抄自投影片的文字，必然對不上。內容是好的，
    #: 只是來源標錯就被退回。新說明給的是可操作的判準（「把你寫的字拿去跟
    #: 兩份材料比對，哪一份重疊多就標哪一份」）而不是語意判斷。
    #:
    #: v4/v5 這兩個號碼是**誤觸**的：改 S4a 的版本時用了未錨定的字串替換
    #: （`prompt_version: str = "vN"` 在兩個類別裡長得一模一樣），把本階段
    #: 一起 bump 了兩次。第二次更糟——它讓我改完 prompt 後版本**沒有變化**，
    #: 快取直接命中，**新 prompt 從未被執行**，量出來卻只當成「改了沒用」。
    #: 差一點就據此做出錯誤結論。見 decisions.md D30。
    #:
    #: v3（2026-08-06）：拆出 S4a 後，本階段不再負責 is_slide 與 slide_text。
    #: **改了 prompt 就必須改這個**——否則舊快取會被當成新結果讀回來。
    #: v0.3 首跑的 07_understanding 是 v1，帶著 30.6% 的圖片錯位。
    prompt_version: str = "v7"
    #: 可將 2–3 個相鄰 segment 併為一次呼叫，但輸出仍逐 segment 分開
    batch_segments: int = 3
    prev_summary_max_chars: int = 200
    max_retries: int = 2
    retry_backoff_sec: float = 4.0
    #: §5.5 #6：本地 fallback 必須是明確開關，且輸出須標記 model_used。
    #: 預設關閉——額度耗盡時停止，不靜默降級。
    allow_local_fallback: bool = False
    local_fallback_model: str = "Qwen/Qwen3-VL-8B-Instruct"


class S4bConfig(StageConfig):
    """以投影片術語為詞庫的事後校正。SDD §4.7b（v0.4 新增）。

    R20 實測：拼音門檻 0.90 下 precision **90%**，抓到 14 個相異錯誤；
    對照組是 VLM 在人工字幕上產生的 7 筆。純本地，不花額度。
    """

    enabled: bool = True
    #: 拼音相似度下限。R20 實測 0.70/0.80/0.90 的 precision 為 50%/79%/90%，
    #: 且 0.90 與 1.00 結果完全相同——0.9 以上的候選實際上都是完全同音。
    #: 誤報集中在 0.70–0.80（`潢房子是`→`同房子是` 這類無意義的窗）。
    #:
    #: **90% 恰好踩在 §5.2 的門檻（術語校正 precision ≥ 0.90）上，沒有餘裕。**
    min_pinyin_similarity: float = 0.90


class S5Config(StageConfig):
    model: str = "gemini:gemini-3.1-flash-lite"
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

    #: **逐支跑 S-1**（v0.4，§4.0）。原本是「每系列抽樣一次」，
    #: 實測同一個播放清單的第 14、27 集換了攝影棚背景，抽樣得到的 profile
    #: 推廣不到同系列其他集。與 S1b 共用抽幀，邊際成本接近零。
    survey_each_video: bool = True

    work_dir: Path = Path("work")
    out_dir: Path = Path("out")
    log_level: str = "INFO"

    s0: S0Config = Field(default_factory=S0Config)
    s1a: S1aConfig = Field(default_factory=S1aConfig)
    s1b: S1bConfig = Field(default_factory=S1bConfig)
    s1c: S1cConfig = Field(default_factory=S1cConfig)
    s3: S3Config = Field(default_factory=S3Config)
    s4a: S4aConfig = Field(default_factory=S4aConfig)
    s4: S4Config = Field(default_factory=S4Config)
    s4b: S4bConfig = Field(default_factory=S4bConfig)
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
