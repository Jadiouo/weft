# weft

把 YouTube 上的中文講經／授課影片系列，批次轉換成可檢索、可溯源的結構化
知識庫，輸出為向量庫可直接匯入的 JSONL。

**規格見 [`SDD.md`](SDD.md)。動工前請先讀完第 5 章「驗證」——它刻意排在
第 6 章「實作」之前。**

## 現況：Phase 0 完成（骨架與驗證框架）

依 SDD §7.1，驗證框架先於任何功能程式碼建立。

| 項目 | 狀態 |
|---|---|
| 專案結構、設定檔、logging | ✅ |
| IR schema（pydantic，`extra=forbid`） | ✅ |
| 合成測試影片產生器 A1–A7 | ✅ 7 支／943 秒／15 秒可重建 |
| §5.3 不變量檢查器（10 條） | ✅ 每條皆有反例測試 |
| §5.2 指標計算 | ✅ |
| §5.4 溯源檢查（防幻覺閘門） | ✅ 正向／反向／具名實體 |
| §5.5 禁止捷徑的機械式護欄 | ✅ 涵蓋 7/13 條（其餘見 known-risks R3） |
| 空的 e2e 測試 | ✅ 跑不通，但存在 |

```
$ pytest -q
13 failed, 160 passed, 4 skipped
```

**這是預期結果**，也正是 §7.1 的完成條件：「跑 pytest 會失敗，但失敗訊息
清楚指出缺少哪些實作」。13 個失敗全部來自尚未實作的階段，每個都會報出
自己是誰、對應 SDD 哪一節、屬哪個 Phase、還缺什麼：

```
階段 S1b 投影片候選幀 尚未實作（SDD §4.3，屬 Phase 1）
  待實作：
    - ffmpeg 每 1 秒抽一幀縮圖
    - speaker/slide 二分類：偵測滿版人臉（本素材硬切、無 PiP）
    ...
```

## 環境

SDD §8 要求兩個 conda 環境。目前只建了 `pipe-cpu`（Phase 0 不需要 GPU）。

```bash
conda create -n pipe-cpu python=3.11 -y
conda run -n pipe-cpu pip install -e ".[dev]"
```

`pipe-gpu` 待 Phase 1 建立，注意 SDD §8 的兩個雷：PyTorch 必須從官方 index
用 pip 裝（Blackwell sm_120），ffmpeg 用系統 apt 版本。

## 用法

```bash
weft synth                      # 產生 A1–A7 合成測試影片
weft prepare <playlist|video>   # S0–S3，本地，不花額度
weft understand                 # S4–S6，消耗 Gemini 額度，額度用盡自動停
weft status                     # 掃描 work/，列出各影片的階段完成狀態
```

`prepare` / `understand` 分離對應 SDD §6.4：前者可先把整個 playlist 處理完
囤在硬碟，後者每天按額度消化，額度重置時工作已就緒。

## 測試

```bash
pytest -q                        # 全部
pytest -m "not gpu and not quota"  # 不需 GPU 與 API 額度
pytest -m synth                  # 只跑合成影片相關（首次會用 ffmpeg 產生）
```

## 文件

- [`SDD.md`](SDD.md) — 規格。唯一權威。
- [`docs/decisions.md`](docs/decisions.md) — 實作時才能決定、且有實測依據的選擇
- [`docs/known-risks.md`](docs/known-risks.md) — 未驗證的假設與何時能驗證

## 給後續實作者（含 AI coding agent）的提醒

SDD §5.5 有一份**禁止捷徑清單**。其中最容易被無意違反的是：

- 不得為了讓測試通過而調低 §5.2 的門檻。門檻在
  `src/weft/validation/thresholds.py`，是設計目標，不是可調參數。
- 不得縮小測試集或移除對抗樣本（A1–A7 為必選）。
- 不得把 §5.3 的 assert 改成 warning 或 log。
- e2e 測試不得用 mock 取代真實模型呼叫。

其中 7 條已由 `tests/test_unit_conventions.py` 自動檢查。
**若你認為某項規定阻礙進度，正確做法是停下來提出討論，而不是繞過。**
