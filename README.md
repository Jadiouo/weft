# weft

把 YouTube 上的中文講經／授課影片系列，批次轉換成可檢索、可溯源的結構化
知識庫，輸出為向量庫可直接匯入的 JSONL。

**規格見 [`SDD.md`](SDD.md)。動工前請先讀完第 5 章「驗證」——它刻意排在
第 6 章「實作」之前。**

---

## 現況

SDD §4 的十個階段全部實作完成。

```
$ pytest -m "not gpu and not quota"
323 passed, 3 skipped
```

3 個 skip 是等真實影片黃金集（§5.1 B）的門檻，**不是假通過**——skip 會
出現在報告裡。

| 階段 | SDD | 狀態 |
|---|---|---|
| S0 取得 | §4.1 | ✅ 真實影片實跑通過 |
| S1a 逐字稿 | §4.2 | ✅ 手動字幕優先，Whisper 備援 |
| S1b 投影片候選幀 | §4.3 | ✅ 合成素材 F1 = 1.000（門檻 0.95）｜⚠️ 真實素材見 R8 |
| S2 OCR | §4.4 | ✅ 直排中文信心 0.98–1.00 |
| S2b 術語詞庫 | §4.4 | ✅ 系列級累積 |
| S2c 術語校正 | §4.5 | ✅ precision = 1.000（門檻 0.90） |
| S3 對齊 | §4.6 | ✅ 合成素材邊界位移 0.0s |
| S4 聯合理解 | §4.7 | ⏳ 已實作，待 `GEMINI_API_KEY` 驗證 |
| S5 全片統整 | §4.8 | ⏳ 同上 |
| S6 渲染 | §4.9 | ✅ 不需 API 的部分已驗證 |

### 驗證框架（SDD 第 5 章）

| 項目 | 狀態 |
|---|---|
| 合成測試影片 A1–A7 | ✅ 7 支／943 秒／17 秒可重建 |
| §5.3 不變量（10 條） | ✅ 每條皆有反例，且在**真實管線輸出**上驗證 |
| §5.2 指標 | ✅ |
| §5.4 溯源檢查 | ✅ 正向／反向／具名實體 |
| §5.5 禁止捷徑護欄 | ✅ 自動檢查 13 條中的 7 條 |
| §5.1(B) 真實影片黃金集 | ❌ **未標註**，工具已備（見下） |

### 需要你決定的事

1. **`docs/known-risks.md` R8：真實素材的畫面結構與 SDD §1.3 不符。**
   實測 zIglvjoU9vo 有 81% 是固定攝影棚機位（講者始終在畫面中），
   而非 §1.3 描述的「全螢幕講者 ↔ 全螢幕投影片硬切」；另有 §1.3 未提及的
   「半透明疊加」模式。這影響 §4.3 的分類設計，**我沒有單方面修改**。

2. **黃金集標註**（§5.1 B，每支約 30 分鐘）。工具見
   `tests/golden/annotate.py`。在 R8 釐清「什麼算一次換頁」之前，標註的
   定義本身還不明確。

3. **Gemini API key**：S4–S6 需要 `GEMINI_API_KEY`（AI Studio free tier）。

---

## 環境

SDD §8 要求兩個 conda 環境，皆已建立：

```bash
# pipe-cpu：S0、S1b、S3(部分)、S6
conda create -n pipe-cpu python=3.11 -y
conda run -n pipe-cpu pip install -e ".[dev]"

# pipe-gpu：S1a、S2、S3、S4 fallback
conda create -n pipe-gpu python=3.11 -y
conda run -n pipe-gpu pip install torch --index-url https://download.pytorch.org/whl/cu128
conda run -n pipe-gpu pip install -e ".[dev,gpu]"
```

SDD §8 的兩個雷都已避開並驗證：PyTorch 從官方 index 裝（`2.11.0+cu128`，
sm_120 實跑通過）、ffmpeg 用系統 apt 版（`/usr/bin/ffmpeg` 6.1.1）。
第三個雷是實作時才發現的：PaddleOCR 必須關閉 oneDNN，且要用 CPU 版
paddle 3.x（見 `docs/decisions.md` D10）。

## 用法

```bash
weft synth                      # 產生 A1–A7 合成測試影片
weft prepare <playlist|video>   # S0–S3，本地，不花額度
weft understand                 # S4–S6，消耗 Gemini 額度，用盡自動停
weft status                     # 掃描 work/，列出各影片的階段完成狀態
```

`prepare` / `understand` 分離對應 SDD §6.4：前者可先把整個 playlist 處理完
囤在硬碟，後者每天按額度消化，額度重置時工作已就緒。

## 測試

```bash
pytest -m "not gpu and not quota"   # 日常（約 45 秒）
pytest -m synth                     # 只跑合成影片相關
pytest -m golden                    # 真實影片黃金集
pytest                              # 全部（需 GPU 與 API key）
```

## 文件

- [`SDD.md`](SDD.md) — 規格。唯一權威。
- [`docs/decisions.md`](docs/decisions.md) — 實作時才能決定、且有實測依據的
  選擇（D1–D13，每則都附實測數據）
- [`docs/known-risks.md`](docs/known-risks.md) — 未驗證的假設與何時能驗證
  （R1–R8，含已解除的）

## 給後續實作者（含 AI coding agent）的提醒

SDD §5.5 有一份**禁止捷徑清單**。最容易被無意違反的是：

- 不得為了讓測試通過而調低 §5.2 的門檻。門檻在
  `src/weft/validation/thresholds.py` 的 `ACCEPTANCE_THRESHOLDS`。
- 不得縮小測試集或移除對抗樣本（A1–A7 為必選）。
- 不得把 §5.3 的 assert 改成 warning 或 log。
- e2e 測試不得用 mock 取代真實模型呼叫。

其中 7 條已由 `tests/test_unit_conventions.py` 自動檢查。
**若你認為某項規定阻礙進度，正確做法是停下來提出討論，而不是繞過。**
