# weft — 影片 → 可溯源知識庫 Pipeline

每個 session 動這個 repo 之前先讀這份文件，以及 `SDD.md` 的 §5（驗證）。

---

## 這個專案在做什麼

把影片（YouTube 播放清單）批次轉成**可檢索、可溯源**的結構化知識庫，
輸出 JSONL 供向量庫匯入。最終去向是使用者的 Obsidian vault
（`~/Documents/library`），但**本專案不做 Obsidian 整合**——見 SDD §1.2。

**v0.5 起主幹改為逐字稿**（投影片降為輔助）。理由與前人工作見
`docs/research/2026-08-08-prior-art-transcript-first.md`。

---

## 環境

```bash
# Python 環境是 conda，不是 .venv
~/miniconda3/envs/pipe-cpu/bin/python     # 主要
~/miniconda3/envs/pipe-gpu/bin/python     # GPU 工作

# 測試。**預設就是離線的**——不需要網路、雲端額度、或本地模型服務。
# 乾淨機器上實測 628 passed / 5 skipped / 0 failed，19 秒。
~/miniconda3/envs/pipe-cpu/bin/python -m pytest tests/ -q

# 需要外部資源的那一層（下載影片 + 本地 ollama）。**手動跑。**
~/miniconda3/envs/pipe-cpu/bin/python -m pytest -m live -q
```

### 兩層測試

| 層 | 依賴 | 何時跑 |
|---|---|---|
| 預設（`pytest tests/ -q`） | 無。ffmpeg 產生的合成素材會快取 | 每次改動 |
| `-m live` | 網路下載 + ollama 服務 | 手動，改動 S0/S4 時 |

預設層排除 `live`、`quota`、`gpu` 三個 marker（設在 `pyproject.toml` 的
`addopts`）。CLI 的 `-m` 會蓋掉它，所以 `pytest -m live` 照樣跑得到。

`tests/test_e2e_offline.py` 是離線 e2e：拿快取的 IR 跑 S6 渲染、§5.4 溯源、
§5.3 不變量、`chunks.jsonl` 契約。**它不 mock 模型**（§5.5 #10）——
它根本不呼叫模型，測的是拿到理解結果之後那一段。

需要真實影片的黃金集測試在沒有 `work/` 時**會 skip 而不是假通過**。

### 目前已知的紅燈（只有一個）

```
tests/test_e2e_pipeline.py::test_slide_classification_on_real_videos
  2FjApOVIbUs[保留集] 0.909 ／ C_CFyilE-ks[調校集] 0.947
  cxrqHABhWOU[調校集] 1.000 ／ zIglvjoU9vo[調校集] 0.864     門檻 0.95
```

**這是刻意留著的真紅燈，不是壞掉的測試**（R26：改 prompt 的措辭已到頂，
再調就是對黃金集過擬合）。把它藏到 marker 後面就變成 §5.2 那四項
「只 assert 常數值」的同類——綠燈製造「有人管」的錯覺。

回來時比對這四個數字：**一樣就是沒退步，變了才要查**。
它需要 `work/` 下的快取產物；乾淨機器上這條會 skip，全套是
**628 passed / 5 skipped / 0 failed**（有 `work/` 時 632 passed / 1 failed）。

### 第二條要盯的：分段（2026-08-09 起）

```
tests/test_e2e_pipeline.py::test_segmentation_beats_not_segmenting_at_all
  cxrqHABhWOU 一刀不切 0.451 / 現行 0.360  贏
  2FjApOVIbUs 一刀不切 0.464 / 現行 0.359  贏
  UiKi5-Arce4 一刀不切 0.467 / 現行 0.562  **輸**
```

**這條是綠的，但它印出來的第三行是輸的。** 那不是被藏起來——
三個數字都釘在測試裡，退步（曾經贏過的變成輸）或漂移超過 0.01 都會紅。

「一刀不切」是免費的下界。**贏不過它就是在做負功**——
R40 就是靠加這條線，發現 Hearst 1997 的原始 α=−0.5 在三支上全輸。
STEM 那支目前仍未修好（票 16），見 known-risks R33。

> **不要用 boundary F1 判斷分段好壞。** 它對過度分割結構上不敏感：
> 同一批改動在 ±10s F1 上 0.429→0.421（幾乎不動），
> 在 WindowDiff 上是 0.529→0.360。見 R37／R40 與
> `docs/research/2026-08-09-prior-art-segmentation-granularity.md`。

> `python` 這個裸指令在這台機器上不存在。用完整路徑或先 activate。

**已知環境問題**：yt-dlp 的 JS challenge。機器上有 node 沒有 deno，
`s0.js_runtimes` 已指定 node，消掉了 "No supported JavaScript runtime" 的警告；
但**實測格式數與解析度完全沒變**（都是 16 個格式、最高 720p）。
要再往上得開 `--remote-components ejs:github`——執行期從 GitHub 下載並
執行腳本，**刻意不開**，見 `docs/FROZEN.md` F10。

---

## 跑任何「改了有沒有效」的量測之前

**S4c 不是確定性的**（`temperature=0.2`、無 seed）。實測同設定重跑五次，
總產出字數 600–1386，**CV 0.34**（known-risks R36）。

> **兩邊都要至少 3 個樣本。** 清 `work/<vid>/07_understanding` 會重新
> 呼叫模型；10 段的影片約 10 分鐘。

分段那一層不受影響——它是在已快取的逐字稿上做確定性計算。

> **而且要驗「它真的做了那件事」的正面訊號**（R37）。實測踩過兩次：
> `>/dev/null 2>&1` 吞掉 `ollama` 沒開的錯誤，量測讀到沒動過的舊檔，
> 三次「完全相同」看起來剛好像我預期的結論。
> 檢查 `10/10 個 segment` 出現、檢查快取檔真的寫出來、失敗就中止。

---

## 這個專案最重要的一條紀律

> **未驗證的假設，和實測過的事實，在文件裡必須長得不一樣。**

這個 repo 的歷史上，同一個錯誤犯過三次，每次都是把假設寫成前提：

| | 從 | 推廣到 | 後果 |
|---|---|---|---|
| v0.2 | §1.3 單支影片的觀察 | 整個系列 | chunk 內容變成攝影棚背板裝飾字，**而所有機械檢查全綠** |
| v0.4 | 第 1 集的 profile | 第 14 集 | 背景換了，分界值失效 |
| R20 | 一次 `initial_prompt` 實驗 | 「解碼層沒用」 | 與 DocWhisper 的 14.3% 改善矛盾，見 research 檔 §4 |
| R26 | 調校集 1.000 | 判準成立 | 保留集揭穿：那是素材的統計巧合 |

**寫任何規格、報告、決策時**：實測過的標出處（`D<n>`、`R<n>`、實驗路徑），
沒實測的明寫「假設，未驗證」。不確定的數字不要寫進表格裡當事實。

`§5.2` 的門檻只有 3/10 真的在驗收——見 D30 與 `DANGLING_THRESHOLDS`。
**「檢查存在」與「檢查有效」是兩件事，而前者會偽裝成後者。**

---

## 文件導覽

| 檔案 | 內容 |
|---|---|
| `SDD.md` | 主規格。**§5 驗證在 §6 實作之前，動手前必讀** |
| `docs/FROZEN.md` | **刻意不做的東西** + 恢復觸發條件。做決定前先看這裡 |
| `docs/decisions.md` | D1–D31 決策記錄，含已作廢的 |
| `docs/known-risks.md` | 已知風險，`DANGLING_THRESHOLDS` 的配套紀錄 |
| `docs/research/` | 前人工作勘查（帶引用） |
| `docs/proposals/` | 提案，含未採納的 |
| `docs/worklog/` | 每日工作日誌 |
| `experiments/r*/REPORT.md` | 每個實驗的量測結果 |
| `.scratch/<feature>/` | 進行中的 spec 與 issue |

---

## 模型配置

**不使用雲端 API（Gemini）**。2026-08-08 的決定，理由與代價見 `docs/FROZEN.md`。
程式碼保留在 `stages/providers.py`，只是不啟用。

全部走本地（ollama）。代價：溯源通過率 0.986 → 0.838，`slide_text` CER 上升。
**這是已知且接受的。**

---

## Agent skills

本 repo 已配置 [mattpocock/skills](https://github.com/mattpocock/skills)。

### Issue tracker

本機 markdown，位於 `.scratch/<feature-slug>/issues/`。見 `docs/agents/issue-tracker.md`。

### Domain docs

單一 context：根目錄 `CONTEXT.md` + `docs/adr/`。見 `docs/agents/domain.md`。

### 常用流程

```
/grill-with-docs  →  /to-spec  →  /to-tickets  →  /implement（每張票 /clear 一次）
```

**不要直接開始寫規格。** 這個 repo 的繞路史全部來自「腦內自問自答，
猜測無聲升格成前提」。先 `/grill-with-docs` 把問題問出來，
`/to-spec` 只做合成、不做訪談。
