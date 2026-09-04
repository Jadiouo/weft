# weft

把 YouTube 上的中文講課／講經影片系列，批次轉成**可檢索、可溯源**的逐字稿
檔案庫——每一段都能回溯到影片的秒數。

> **v1 已封版（2026-09-04）。** 交付的是**逐字稿檔案庫**，不是知識庫。
> 這個區別是刻意的，理由見 [`docs/v1-scope.md`](docs/v1-scope.md)。

---

## 這個東西實際做了什麼

先講**它沒做的**，因為那比較容易被誤會：

- **它沒有改進語音辨識。** 31 支素材裡有 27 支的逐字稿文字是
  **未經任何修改的 Whisper 輸出**（實測 `corrections` = 0）。
- **它不是通用工具。** v1 的契約是產物導向——這 31 支的產出。
  它能跑別的播放清單，但沒有為此驗收過。
- **它不做 Obsidian 整合**，雖然產出最終要進 vault。

它做的是這些：

| | 為什麼 `yt-dlp \| whisper` 不等於它 |
|---|---|
| **人工字幕優先** | 31 支裡 4 支有人工字幕。直接餵 Whisper 會把乾淨的字幕重聽一遍，**變差** |
| **主題分段** | Whisper 給的是 cue（一句一句）。weft 產出 1,459 個**主題段落**，並且拿「一刀不切」當免費下界量過——**三支黃金集裡贏兩支、輸一支**（見下）|
| **可續跑的階段機** | 每個階段有冪等鍵。參數變了會正確重跑，中斷可續。26 支無人跑 95 分鐘 |
| **會說實話的回報** | 批次結束時印的數字必須兩種結果都印對過。曾經有一次 23/26 下載失敗卻印「全部成功」——那條訊息現在有 5 條測試守著 |
| **驗證框架** | 649 條測試、9 支合成影片、5 支人工標註的真實黃金集 |

### 實際規模

```
31 支影片 · 428,725 字逐字稿 · 48,393 個時間戳 · 1,459 個主題分段
```

---

## 一個禮拜換到的東西：Whisper 在中文講課素材上的邊界

27 支 Whisper 逐字稿，全部量過、全部人讀過。**這張表的用處是省下別人的
一個禮拜**——它說的不是「Whisper 好不好」，而是**它的失敗有固定形狀**。

| **做得到** | 證據 |
|---|---|
| 連貫、切題、看得懂的中文 | 27/27 支人讀確認 |
| 沒有幻覺、重複塌陷、語言切換 | 字/秒 4.16–5.27（**全距很窄**）；最大 6-gram 重複 2–15 次，是講課口頭禪 |
| 英文術語大致保住 | `Feedback`／`full rank`／`singularity`／`Jacobian` |
| 時間戳完整 | 48,393 個 cue，31 支無缺 |

| **做不到** | 證據 |
|---|---|
| **標點** | 全形標點密度 **0.00%–0.17%**，等於零 |
| **領域專有名詞** | 觀察到的錯誤**全部集中在這一類**（見下）|

| **沒測過——不要當作知道** | |
|---|---|
| 說話人分離 | 素材全是單一講者，沒有機會測 |
| 非中文素材、短影片、非講課體裁 | 全部是 15–88 分鐘的中文講課 |

錯誤長這樣，**全是專有名詞被聽成音近的非術語**：

| 逐字稿 | 應為 | | 逐字稿 | 應為 |
|---|---|---|---|---|
| `ControlRaw` | Control **Law** | | `Homogenetic Transformation` | Homogene**ous** |
| `卡住的關係` | 卡氏座標系 | | `determined j` | determinant J |
| `彈音係數` | 彈簧係數 | | `Effective Inertial` | Effective Inert**ia** |
| `Learning my demonstration` | Learning **by** demonstration | | `你的城市` | 你的**程式** |

> **失敗有固定形狀，就代表它可以被針對性地修。**
> 而「怎麼修」也已經量過了——**已實測可用的是事後校正（precision 100%），
> 不是解碼時注入**（全域 `initial_prompt` 實測沒用，逐段 hotwords 未測）。
> 完整的流程圖與四處修正見 [`docs/v1-scope.md`](docs/v1-scope.md) §5.1。

---

## 這個 repo 真正的產出，可能是它記下的「什麼行不通」

開發實際發生在 **7 個工作天**（2026-08-04 至 09-04，100 個 commit
分佈在 7 個日期上），留下 **35 個實驗、27 份完整報告**、
**D1–D36 的決策紀錄**（`docs/decisions.md`）與
**R1–R39 的風險紀錄**（`docs/known-risks.md`）。裡面大部分是**負面結果**，
每一條都有實測，而它們不會出現在任何論文裡：

| | 實測結論 | 出處 |
|---|---|---|
| **本地 LLM 的跑次變異** | 同設定重跑五次，總產出字數 600–1386，**CV 0.34** | 風險 R36、`experiments/r44_run_variance/` |
| **貪婪解碼救不了** | `temperature=0 + top_k=1` 在**單一 process 內**確定，跨 process 無效；`seed` 完全沒有作用 | `experiments/r46_determinism/` |
| **不要用 boundary F1 評分段** | 同一批改動在 ±10s F1 上 0.429→0.421（幾乎不動），在 WindowDiff 上是 0.529→0.360 | 風險 R30、`experiments/r37_segmentation_tolerance/` |
| **Hearst 1997 的原始參數輸給不分段** | 加一條「一刀不切」的免費下界，才發現 α=−0.5 在三支素材上**全輸** | 風險 R33、`experiments/r40_granularity/` |
| **溯源檢查量錯了東西** | §5.4 量的是 Source Faithfulness（照著來源講了沒），不是 World Factuality（講的是不是真的）。Whisper 把「座標」聽成「做標」而模型自己改對了，**檢查判它溯不到——我們在懲罰模型做對事** | `experiments/r39_homophone_diagnosis/` §3.1、`docs/v1-scope.md` §5.1 |
| **「沒有錯誤訊息」不等於「成功了」** | `>/dev/null 2>&1` 吞掉 ollama 沒開的錯誤，量測讀到沒動過的舊檔，三次「完全相同」**剛好像我預期的結論** | 風險 R37 |
| **糾錯掛錯了層** | 它掛在「理解」，而它修的是「逐字稿」。實測：人工字幕 4 支修了 16 筆，Whisper 27 支 **0 筆**——方向完全反了 | 風險 R39 |

> **編號有兩套，會撞號。** `docs/known-risks.md` 的 `R<n>` 與
> `experiments/r<n>_*/` 的 `r<n>` 是**各自獨立**的序號——
> 例如風險 R39 是「糾錯掛錯層」，而實驗 r39 是同音字診斷。
> 引用時請連目錄名一起寫。
---

## 這個專案最重要的一條紀律

> **未驗證的假設，和實測過的事實，在文件裡必須長得不一樣。**

這個 repo 的歷史上，同一個錯誤犯過四次，每次都是把假設寫成前提：

| | 從 | 推廣到 | 後果 |
|---|---|---|---|
| v0.2 | 單支影片的觀察 | 整個系列 | chunk 內容變成攝影棚背板裝飾字，**而所有機械檢查全綠** |
| v0.4 | 第 1 集的 profile | 第 14 集 | 背景換了，分界值失效 |
| R20 | 一次 `initial_prompt` 實驗 | 「解碼層沒用」 | 與已發表的 14.3% 改善矛盾，後由 D33 改寫 |
| R26 | 調校集拿到 1.000 | 「判準成立」 | 保留集揭穿：那是素材的統計巧合 |

所以：**實測過的標出處**（`D<n>`、`R<n>`、實驗路徑），
**沒實測的明寫「假設，未驗證」**。不確定的數字不寫進表格裡當事實。

### 一個具體的例子

測試套件裡有**一條刻意留著的紅燈**：

```
tests/test_e2e_pipeline.py::test_slide_classification_on_real_videos
  2FjApOVIbUs[保留集] 0.909 ／ C_CFyilE-ks[調校集] 0.947
  cxrqHABhWOU[調校集] 1.000 ／ zIglvjoU9vo[調校集] 0.864     門檻 0.95
```

**它不是壞掉的測試。** 改 prompt 的措辭已經到頂，再調就是對黃金集過擬合。
把它藏到 marker 後面就會變成「只 assert 常數值」的同類——
**綠燈製造「有人管」的錯覺**。回來時比對這四個數字：一樣就是沒退步。

同理，分段那條測試**是綠的，但它印出來的第三行是輸的**：

```
tests/test_e2e_pipeline.py::test_segmentation_beats_not_segmenting_at_all
  cxrqHABhWOU  一刀不切 0.451 / 現行 0.360   贏
  2FjApOVIbUs  一刀不切 0.464 / 現行 0.359   贏
  UiKi5-Arce4  一刀不切 0.467 / 現行 0.562   **輸**    ← STEM 素材
```

（WindowDiff，**越低越好**。）「一刀不切」是免費的下界——
**贏不過它就是在做負功**。輸的那支是 STEM 中英混合素材，**目前仍未修好**，
硬需求是再標兩支黃金集。三個數字都釘在測試裡，退步（曾經贏的變成輸）
或漂移超過 0.01 都會紅。

---

## 跑起來

```bash
# 需要 Python ≥ 3.11 與 ffmpeg。
pip install -e ".[dev]"

# 沒有人工字幕的影片需要 faster-whisper（含 torch，建議獨立環境）：
pip install -e ".[gpu]"
```

> 這個 repo 沒有 `environment.yml`。開發機上用的是兩個 conda 環境
> （`pipe-cpu` 主要、`pipe-gpu` 跑 Whisper），細節見 [`CLAUDE.md`](CLAUDE.md)。

```bash
# 抓取 + 逐字稿 + 分段（S0–S3）。這是 v1 的交付範圍。
weft -c configs/local.yaml prepare "<playlist-or-video-url>"

# 理解 + 溯源 + 渲染（S4–S6）。**不在 v1 交付範圍內**，需要本地 ollama。
weft -c configs/local.yaml understand <video_id>
```

沒有人工字幕的影片需要 `faster-whisper`（`[gpu]` 那組相依）。
**沒裝時它不會默默降級跳過**——會直接說缺什麼、以及該換哪個環境。

### 測試

```bash
# 預設層。**離線** —— 不需要網路、雲端額度、或本地模型服務。
pytest tests/ -q
#   乾淨機器：642 passed / 8 skipped / 0 failed，19 秒
#   有 work/ 快取：649 passed / 1 failed（那條紅燈是刻意的，見上）

# 需要外部資源的那一層（下載 + 本地 ollama）。手動跑。
pytest -m live -q
```

需要真實影片的測試在沒有 `work/` 時**會 skip 而不是假通過**。

---

## 架構

```
S0  取得      yt-dlp：影片 + metadata + 人工字幕（若有）
S1a 逐字稿    人工字幕優先，否則 faster-whisper
S1b 候選幀    ffmpeg 抽幀 → 靜止區段偵測
S1c 去重      感知雜湊分群
S3  對齊      逐字稿 ↔ 投影片時間軸
────────────── 以上是 v1 的交付範圍 ──────────────
S4a 投影片理解  VLM 判斷「這張是不是投影片」
S4  聯合理解    逐段產生結構化 block
S5  全片統整
S6  渲染        chunks.jsonl（供向量庫匯入）
```

**v0.5 起主幹改為逐字稿**，投影片降為輔助。

模型全部走本地（ollama）。雲端路徑（Gemini）的程式碼保留在
`stages/providers.py` 但**不啟用**——理由與代價見
[`docs/FROZEN.md`](docs/FROZEN.md)。

---

## 文件

| 檔案 | 內容 |
|---|---|
| [`docs/v1-scope.md`](docs/v1-scope.md) | **v1 交付什麼、怎麼驗收、什麼推到 v2。第一件要讀的** |
| [`SDD.md`](SDD.md) | 主規格。**§5 驗證排在 §6 實作之前，這是刻意的** |
| [`docs/FROZEN.md`](docs/FROZEN.md) | **刻意不做的東西** + 恢復觸發條件 |
| [`docs/decisions.md`](docs/decisions.md) | D1–D36，含已作廢的 |
| [`docs/known-risks.md`](docs/known-risks.md) | R1–R39 |
| [`docs/research/`](docs/research/) | 前人工作勘查（帶引用）|
| [`docs/worklog/`](docs/worklog/) | 每日工作日誌 |
| [`experiments/r*/REPORT.md`](experiments/) | 每個實驗的量測結果，**含失敗的** |
| [`CLAUDE.md`](CLAUDE.md) | 給 AI agent 的專案指引 |

---

## 下一步（v2）

v1 的產出是**給 v2 重做用的原料**。方向已經定了，而且每一步都有量測撐著：

1. **建領域詞表**，不是完整語料庫——錯誤幾乎全是專有名詞（見上表），
   而詞表便宜、語料庫貴一個數量級
2. **詞源順位：影片標題（零成本、31/31 已有）→ 投影片 `slide_text`
   （需要對 3,134 張跑一次 VLM）→ 爬蟲（第三順位）**。
   不要用粗逐字稿去找爬蟲目標——**它裡面錯的正好就是你要搜的那些詞**
3. **用 S4b 事後校正**（`stages/lexicon.py`，已實作，實測 precision 100%）。
   解碼時注入是**未驗證**的：全域 `initial_prompt` 實測沒用，
   逐段 hotwords 沒測過——**要試就明寫「假設，未驗證」**
4. 才做成 RAG 或訓練資料

**要防的陷阱**：不能用同一個先驗庫既修又驗——那就回到「拿測驗的結果
當它自己的先驗」。要一開始就切出 held-out。

完整流程圖與四處修正見 [`docs/v1-scope.md`](docs/v1-scope.md) §5.1。

---

## 授權與素材版權

程式碼與文件採 [MIT](LICENSE)。

**授權不涵蓋管線處理的影片素材**——那些是各自著作權人的作品。
版控裡**不含任何影片、音訊或完整逐字稿**：`work/` 與 `out/` 都在
`.gitignore` 裡，`tests/golden/` 只有人工標註與公開的影片標題與 URL，
`docs/` 裡的短句引用是為了記錄具體的 ASR 錯誤。

---

## English

**weft** turns Chinese-language YouTube lecture playlists into a traceable
transcript archive — 31 videos, 428k characters, 1,459 topical segments,
every segment linked back to a timestamp.

It does **not** improve speech recognition: 27 of 31 transcripts are
unmodified Whisper output. What it adds over `yt-dlp | whisper` is manual-caption
preference, topic segmentation validated against a "no cuts at all" baseline,
a resumable stage machine with idempotency keys, and batch reporting that has
been verified to report failure correctly (not just success).

Its most transferable output is probably the **negative results** recorded in
`docs/known-risks.md` — measured findings about local-LLM run-to-run variance
(CV 0.34), the session-scoped nature of greedy decoding, why boundary F1 is
structurally insensitive to over-segmentation, and why source-faithfulness
checks penalise a model for correcting an ASR error.

Docs are in Chinese.
