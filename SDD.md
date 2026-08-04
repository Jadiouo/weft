# SDD：講經影片 → 向量知識庫 Pipeline

> 版本 0.1 ｜ 目標讀者：實作者（人類或 AI coding agent）
> **本文件第 5 章「驗證」在第 6 章「實作」之前。任何實作動作開始前必須先讀完第 5 章。**

---

## 1. 目標與非目標

### 1.1 目標

把 YouTube 上的中文講經／授課影片系列，批次轉換成**可檢索、可溯源的結構化知識庫**，輸出為向量庫可直接匯入的 JSONL。

具體成功條件：

1. 給一個 YouTube playlist URL，能無人值守地跑完整個系列（受每日 API 額度限制，跨日續跑）。
2. 輸出的每一個 chunk 都自足（不依賴前後文即可理解）、帶時間戳、可回溯到影片的具體秒數。
3. 投影片上的內容以**文字**形式進入知識庫（因為 RAG 讀不到圖），且保留版面所隱含的結構關係。
4. 逐字稿中的專有術語被投影片文字校正過。

### 1.2 非目標（明確排除）

- **不做 UI**。CLI + 設定檔。
- **不做 Obsidian 整合**。輸出 JSONL，使用者自行後處理。
- **不做即時／串流處理**。批次離線。
- **不追求 SOTA 偵測精度**。這是個人知識庫工具，不是論文。錯了可以重跑。
- **不做多模態向量檢索**（CLIP embedding）。v1 只做文字檢索。
- **不做跨影片概念圖 / concept registry**。推遲到 v2（見 §7.5）。

### 1.3 目標素材的特性（已確認）

以 `zIglvjoU9vo`（41:59，道教經典講解）為代表：

| 特性 | 狀態 | 對設計的影響 |
|---|---|---|
| 畫面切換 | **硬切**：全螢幕講者 ↔ 全螢幕投影片 | 不需要 ROI 裁切；需要 speaker/slide 二分類 |
| 字幕 | **播放器渲染**（CC 可關閉） | yt-dlp 取得乾淨影片 + 免費字幕軌；**不需要遮罩模組** |
| 雷射筆 | 有，小紅點會移動 | 降解析度 + 高斯模糊即可壓制 |
| 投影片內容 | 密集中文、含**文言文**、含**直排文字**、版面帶語意（箭頭、雙欄、色彩編碼） | Tesseract 不可用；VLM 必須輸出結構化版面描述 |
| 純講者時段 | 大量存在（無投影片） | IR 必須能表達 `mode: speaker_only` |
| 內容層次 | 經文原文 / 白話解說 / 口頭延伸 三種並存 | chunk 需帶 `content_type` 供分別檢索 |

---

## 2. 系統概觀

### 2.1 核心設計原則

**原則一：拆時間軸，不拆模態軸。**
逐字稿與投影片的「提取」可以分開平行；但「理解」必須是聯合的。分別理解兩個模態再合併，會在壓縮階段丟掉對齊所需的資訊，且不可逆。降低單次工作量的方法是**縮小時間範圍**（一次一個 segment），不是拆開模態。

**原則二：本地扛量，雲端扛判斷。**
所有高頻、機械性的工作（ASR、OCR、embedding、CV）在本地 GPU 執行，不受額度限制。只有需要跨模態判斷力的那一步呼叫 Gemini API。

**原則三：不花額度的工作先跑滿，花額度的工作按日消化。**
S0–S3 是 producer，可以先把整個 playlist 處理完囤在硬碟；S4 是 consumer，每天按額度消化 buffer。額度重置時工作已就緒，一秒不浪費在 ffmpeg 上。

**原則四：每個階段落地成檔案，可獨立重跑。**
批次跑數十支影片時，中途失敗是常態而非例外。

### 2.2 資料流

```
S0  取得        yt-dlp → 影片 + 字幕軌 + metadata
                 │
      ┌──────────┴──────────┐         ← 平行，皆為本地、無額度限制
      ▼                     ▼
S1a 逐字稿              S1b 投影片候選幀
    YouTube 字幕優先        抽幀 → speaker/slide 分類
    Whisper 補強/驗證       → HMM 換頁偵測 → 取段末幀
      │                     │
      │                     ▼
      │                 S2  OCR + 術語萃取
      │                     PaddleOCR-VL（本地）
      │                     │
      │                     ▼
      │                 S2b 術語詞庫
      │                     │
      └──────────┬──────────┘
                 ▼
S2c 逐字稿術語校正      用詞庫修正 ASR 錯字
                 │
                 ▼
S3  對齊              時間戳粗切 → embedding 邊界吸附 → segments
                 │
                 ▼
S4  聯合理解  ★★★    per segment：投影片圖 + 校正後逐字稿 + 前段摘要
                       → Gemini（唯一花額度的階段）
                 │
                 ▼
S5  全片統整          只讀 S4 輸出，不再看圖
                 │
                 ▼
S6  渲染              chunks.jsonl（產品）+ debug markdown（人工抽檢用）
```

### 2.3 模型分工

| 用途 | 模型 | 位置 | 理由 |
|---|---|---|---|
| ASR | faster-whisper large-v3 | 本地 GPU | 量最大；本地品質不輸 API |
| 投影片 OCR | PaddleOCR-VL | 本地（CPU 可） | 中文/直排強項；免費無上限；用於建詞庫 |
| Embedding | BGE-M3（或同級多語模型） | 本地 GPU | 對齊與向量化都高頻，絕不打 API |
| **聯合理解** | **Gemini Flash-Lite（API key）** | **雲端** | 文言文＋直排＋版面語意，小模型差距最大 |
| 理解降級路徑 | Qwen3-VL 8B | 本地 GPU | 額度耗盡時的 fallback |

**額度來源明確為 Gemini API key（AI Studio 取得，free tier）。訂閱額度（AI Pro/Ultra）在架構上無法接入程式化呼叫，不得嘗試繞道。**

VRAM 規劃（RTX 5070 Ti，16GB）：Whisper large-v3 與 Qwen3-VL 8B **不可同時常駐**，必須分階段載入／卸載。這在分階段設計下是自然的。

---

## 3. 中介表示（IR）Schema

IR 是整個系統的契約。所有階段圍繞它。

### 3.1 檔案佈局

```
work/
  {video_id}/
    00_meta.json           S0
    01_video.mp4           S0
    01_captions.vtt        S0（若有）
    02_frames/             S1b 抽出的縮圖（可清理）
    02_candidates.json     S1b
    03_slides/             S1b 選定的投影片圖（保留）
      slide_001.png ...
    03_ocr.json            S2
    04_lexicon.json        S2b
    05_transcript.json     S1a + S2c
    06_segments.json       S3
    07_understanding/      S4（每 segment 一檔，便於斷點續跑）
      seg_001.json ...
    08_video.json          最終 IR
    state.json             階段完成狀態與參數 hash
out/
  chunks.jsonl             S6 產品
  debug/{video_id}.md      S6 人工抽檢用
  quota.db                 額度帳本
```

### 3.2 Slide 物件

```json
{
  "slide_id": "slide_005",
  "image_path": "03_slides/slide_005.png",
  "t_first_seen": 748.3,
  "t_last_seen": 845.1,
  "is_progressive_final": true,
  "build_frames": [750.0, 762.4, 771.9],
  "ocr_text": "太上老君內觀經\n天地媾精，陰陽布化...",
  "ocr_confidence": 0.87,
  "layout_description": null
}
```

- `is_progressive_final`：此圖是否為逐條動畫段落的最後一幀（內容最完整）。
- `build_frames`：該段落中被合併掉的中間幀時間戳，保留供 debug。
- `layout_description`：由 S4 的 VLM 填入的結構化版面描述。

### 3.3 Segment 物件

```json
{
  "segment_id": "zIglvjoU9vo#007",
  "video_id": "zIglvjoU9vo",
  "t_start": 748.3,
  "t_end": 845.1,
  "mode": "slide",
  "slide_ref": "slide_005",
  "transcript_raw": "...時運開始進入...",
  "transcript_corrected": "...識蘊開始進入...",
  "corrections": [
    {"from": "時運", "to": "識蘊", "source": "slide_005", "method": "lexicon", "score": 0.91}
  ],
  "boundary_method": "semantic_snap",
  "boundary_shift_sec": -6.2,
  "understanding": { ... }
}
```

- `mode`：`"slide"` ｜ `"speaker_only"` ｜ `"transcript_only"`（降級模式，見 §4.7）
- `boundary_shift_sec`：語意吸附相對於純時間戳切點的位移，供驗證用。

### 3.4 Understanding 物件（S4 產出）

```json
{
  "summary": "本段解說《一月為胞，精血凝也》...",
  "layout_description": "上方為四張胚胎顯微照片（第一至第四周），下方紫底區塊為經文引文與白話解說。",
  "content_blocks": [
    {
      "type": "經文原文",
      "text": "一月為胞，精血凝也。",
      "provenance": {"kind": "slide_ocr", "ref": "slide_005"}
    },
    {
      "type": "白話解說",
      "text": "父母陰陽和合後，受精卵一個月生長成為「胞」，是父精母血的凝聚物。",
      "provenance": {"kind": "slide_ocr", "ref": "slide_005"}
    },
    {
      "type": "口頭延伸",
      "text": "講者以簽約作比喻說明識蘊進入的時機。",
      "provenance": {"kind": "transcript", "ref": "748.3-845.1"}
    }
  ],
  "terms": ["識蘊", "胞", "精血"],
  "unverified_claims": []
}
```

**`content_blocks[].type` 的封閉列舉**：`經文原文` / `白話解說` / `圖表描述` / `口頭延伸`。
**`provenance` 為必填欄位，不得為 null。** 見 §5.4。

### 3.5 輸出 Chunk（S6，JSONL 每行）

```json
{
  "id": "zIglvjoU9vo#007#b02",
  "text": "父母陰陽和合後，受精卵一個月生長成為「胞」...",
  "metadata": {
    "video_id": "zIglvjoU9vo",
    "series_id": "PL37L2Y8jeuGDeSRi5ntD_gK1pBM8KlSkv",
    "video_title": "...",
    "episode_index": 3,
    "t_start": 748.3,
    "t_end": 845.1,
    "url": "https://www.youtube.com/watch?v=zIglvjoU9vo&t=748s",
    "content_type": "白話解說",
    "slide_ref": "slide_005",
    "terms": ["識蘊", "胞"],
    "provenance_kind": "slide_ocr"
  }
}
```

**Chunk 自足性要求**：`text` 欄位不得包含「如上圖」「前面提到的」等指涉性語句，也不得包含 markdown 裝飾。若原始表述有指涉，S4 必須在生成時展開。

---

## 4. 各階段介面契約

每個階段定義：輸入、輸出、冪等鍵、失敗行為。

### 4.1 S0 — 取得

- **輸入**：video_id 或 playlist_id
- **輸出**：`00_meta.json`、`01_video.mp4`、`01_captions.vtt`（若存在）
- **實作**：yt-dlp。字幕優先序：手動字幕 > 自動字幕 > 無
- **冪等鍵**：`video_id`
- **失敗行為**：影片不可用（私人/刪除）→ 記入 skip list，繼續下一支
- **注意**：確認下載的影片**不含硬燒字幕**（已驗證為播放器字幕）

### 4.2 S1a — 逐字稿

- **輸入**：`01_video.mp4`、`01_captions.vtt`
- **輸出**：`05_transcript.json`（含 segment 級時間戳）
- **策略**：
  1. 若有**手動**字幕 → 直接採用，標記 `source: "manual_caption"`
  2. 否則跑 faster-whisper large-v3（`language="zh"`, `initial_prompt` 帶入系列術語詞庫若已存在）→ `source: "whisper"`
  3. 若有**自動**字幕且也跑了 Whisper → 兩者都存，`source: "whisper"` 為主，自動字幕存為 `alt` 供交叉檢查
- **冪等鍵**：`video_id + whisper_model + params_hash`
- **失敗行為**：Whisper OOM → 降低 batch size 重試一次 → 仍失敗則標記影片為 `failed`，繼續下一支

### 4.3 S1b — 投影片候選幀

這是移植 vid2slides 演算法的部分。**移植演算法，不 fork 該 repo**（該 repo 為 2020 年程式碼，conda 環境已不可用）。

- **輸入**：`01_video.mp4`
- **輸出**：`02_candidates.json`、`03_slides/*.png`
- **流程**：
  1. ffmpeg 每 1 秒抽一幀縮圖（可設定）
  2. **speaker/slide 二分類**：偵測滿版人臉 → 標記為 speaker 幀（本素材為硬切，無 PiP，故此步足夠）
  3. 對 slide 幀：降解析度至短邊 ~180px + 高斯模糊（壓制雷射筆紅點與壓縮雜訊）
  4. **HMM 換頁偵測**：以「投影片會停留一段時間」為先驗建模，避免手調門檻
  5. **逐條動畫合併**：若連續段落之間為單調的內容增加（新段落的 OCR 文字包含舊段落的），視為同一頁的 build，合併並**取最後一幀**
  6. 每個穩定段落輸出一張 `slide_NNN.png`
- **冪等鍵**：`video_id + fps + detector_params_hash`
- **失敗行為**：偵測到 0 張投影片 → 標記 `mode: transcript_only`，**不中斷**，繼續走 S1a → S3 → S4（無圖模式）

### 4.4 S2 — OCR 與術語詞庫

- **輸入**：`03_slides/*.png`
- **輸出**：`03_ocr.json`、`04_lexicon.json`
- **實作**：PaddleOCR-VL，本地執行，無額度限制
- **詞庫萃取**：
  - 從 OCR 文字中抽出候選術語（2–6 字的專有名詞、四字詞、書名號與括號內文字）
  - **詞庫 scope 為系列級**（`series_id`），不是單片級——同系列後續影片可直接受惠
  - 詞庫可累積，新影片的術語 append 進去
- **冪等鍵**：`slide_set_hash + ocr_model`
- **注意**：OCR 對直排文字可能出錯，此處**不求完美**——詞庫只需要涵蓋高頻術語即可，精確版面理解由 S4 的 VLM 負責

### 4.5 S2c — 逐字稿術語校正

**這是本專案品質的關鍵路徑，不是可選功能。**

- **輸入**：`05_transcript.json`、`04_lexicon.json`
- **輸出**：`05_transcript.json`（就地更新，加入 `corrected` 與 `corrections` 欄位）
- **方法**：對逐字稿中每個候選片段，與詞庫條目做**拼音/字形相似度**比對（中文 ASR 錯誤多為同音或近音），超過門檻則替換
- **約束**：
  - 每次替換必須記錄 `{from, to, source, method, score}`
  - **只在時間上鄰近的投影片詞庫中比對**（±N 個 segment），避免用第 40 分鐘的術語去改第 5 分鐘的話
  - **保留原文**：`transcript_raw` 永不覆寫
- **失敗行為**：詞庫為空 → 跳過此階段，`transcript_corrected = transcript_raw`

### 4.6 S3 — 對齊

- **輸入**：`05_transcript.json`、`02_candidates.json`
- **輸出**：`06_segments.json`
- **流程**：
  1. 以投影片切換時間戳做**粗切**
  2. 對每個邊界，取前後 ±20 秒的句子，計算其 embedding 與前後兩張投影片 OCR 文字的相似度
  3. 將邊界**吸附**到相似度轉折點
  4. 純講者時段自成 segment，`mode: "speaker_only"`
- **關鍵約束**：吸附範圍硬限制在 ±20 秒內。這一步**不呼叫 LLM**，避免與 S4 形成循環依賴
- **冪等鍵**：`transcript_hash + candidates_hash + embedding_model`

### 4.7 S4 — 聯合理解（唯一花額度的階段）

- **輸入**（per segment）：
  - 投影片圖（`mode == "slide"` 時）
  - 該 segment 的 `transcript_corrected`
  - 前一 segment 的 `summary`（維持連貫性，不超過 200 字）
  - 系列術語詞庫（截取相關部分）
- **輸出**：`07_understanding/seg_NNN.json`
- **模型**：Gemini Flash-Lite（可設定），structured output（JSON schema）
- **批次策略**：可將 2–3 個相鄰 segment 合併為一次呼叫以節省額度，但**輸出仍須逐 segment 分開**
- **降級模式**：
  - `mode == "speaker_only"` → 不送圖，純文字理解
  - `mode == "transcript_only"` → 全片無投影片，退化為逐字稿結構化
  - 額度耗盡 → 停止並記錄進度，**不改用本地模型偷渡**（本地 fallback 須為明確的設定開關，見 §5.5）
- **冪等鍵**：`segment_id + prompt_version + model`
- **失敗行為**：單一 segment 失敗 → 重試 2 次（指數退避）→ 仍失敗則標記 `understanding: null`，繼續下一個 segment

**Prompt 必須明確要求的事項**：
1. 輸出結構化版面描述（箭頭指向、雙欄對應、色彩編碼的意義），不只是文字
2. 每個 `content_block` 標註 `type` 與 `provenance`
3. 展開所有指涉性語句（「這個式子」→ 指明是哪個）
4. **不得推測畫面上與逐字稿中都沒有的資訊**

### 4.8 S5 — 全片統整

- **輸入**：`07_understanding/*.json`（**不再讀圖**）
- **輸出**：`08_video.json` 的頂層欄位（TL;DR、術語總表、章節結構）
- **成本**：1–2 次呼叫

### 4.9 S6 — 渲染

- **輸入**：`08_video.json`
- **輸出**：
  - `out/chunks.jsonl` — 產品
  - `out/debug/{video_id}.md` — 人工抽檢用，含內嵌圖片與可點的時間戳連結
- **chunk 切分規則**：一個 `content_block` = 一個 chunk。若單一 block 超過 800 字則按句切分，但**每個切片都要複製完整 metadata**

---

## 5. 驗證

> **本章在實作章節之前，且為實作的前置條件。**
> 驗證框架必須先於功能程式碼建立。§5.5 的禁止事項為硬性規定。

### 5.1 測試資料

**（A）合成影片** — 主要驗證手段

用 ffmpeg 合成，切換時間點由自己指定，ground truth 精確到毫秒、免費、可重現。

必須包含以下**對抗樣本**，缺一不可：

| 編號 | 樣本 | 期望行為 |
|---|---|---|
| A1 | 標準整頁換頁，每頁停留 30–120 秒 | 全部正確偵測 |
| A2 | **逐條動畫**，一頁分 6 次疊加出現 | 偵測為 **1 頁**，取最後一幀 |
| A3 | **純講者無投影片**（靜態人像 + 輕微晃動） | 偵測為 **0 頁** |
| A4 | **移動紅點**疊加在靜態投影片上 | 偵測為 **1 頁**（紅點不觸發換頁） |
| A5 | 投影片內嵌播放的短影片 | 不切成數十頁 |
| A6 | 講者頁與投影片頁**交錯**出現 | 分類正確，投影片頁數正確 |
| A7 | **回放**：講者退回前一頁再前進 | 偵測為 3 次切換（非 2 次） |

**（B）真實影片黃金集** — 現實校準

手工標註 3 支真實影片（含目標 playlist 的至少 2 支）的換頁時間點。標註成本約每支 30 分鐘，一次性投資。

### 5.2 量化門檻

**門檻寫死在測試程式碼中，不得在測試失敗時調整。**

| 指標 | 對象 | 門檻 | 備註 |
|---|---|---|---|
| 換頁偵測 boundary F1（容忍 ±2 秒） | 合成影片 | **≥ 0.95** | 環境完全可控，應接近滿分 |
| 換頁偵測 boundary F1（容忍 ±2 秒） | 真實影片 | **≥ 0.75** | 參考：文獻中 naive frame diff 約 0.50–0.60，好方法約 0.81+ |
| speaker/slide 分類 accuracy | 兩者 | **≥ 0.95** | 硬切素材，這題應該很簡單 |
| 逐條動畫合併正確率（A2） | 合成 | **= 1.00** | 這是設計目標，不容失敗 |
| 術語校正 precision | 黃金集 | **≥ 0.90** | 寧可漏改，不可亂改 |
| 術語校正 recall | 黃金集 | 記錄但不設硬門檻 | 首版求穩 |
| 對齊邊界誤差中位數 | 黃金集 | **≤ 5 秒** | |
| 溯源通過率 | 全部 | **≥ 0.95** | 見 §5.4 |

### 5.3 機械式不變量（assert，不需人工判讀）

每次跑完必須全數通過，任一失敗即中止並報錯：

1. segments 時間區間**互不重疊**
2. segments 的聯集**等於影片全長**（容忍 ±1 秒）
3. 每一句逐字稿**恰好**被指派到一個 segment
4. 所有 `slide_ref` 指向存在的 slide 物件
5. 所有 `image_path` 對應的檔案**實際存在**且可開啟
6. 所有時間戳**單調遞增**且在 `[0, duration]` 範圍內
7. 每個 `content_block` 的 `provenance` **非 null**
8. 每個輸出 chunk 的 `metadata` 欄位**完整**（無缺欄、無 null）
9. `transcript_raw` 在任何階段後**未被修改**（hash 比對）
10. `corrections` 中每一筆的 `from` 字串**實際出現**在 `transcript_raw` 中

### 5.4 溯源檢查（防幻覺閘門）

這是防止 LLM 編造內容的**唯一有效機制**，必須實作。

**正向檢查**：對每個 `content_block`，取其 `provenance` 指向的來源文字（slide OCR 或該時段逐字稿），計算 block 內容與來源的相似度。

- 相似度 < 門檻 → 標記為 `unverified`
- 每支影片 `unverified` 比例 > 5% → 整支標記為 `needs_review`，不進入 `chunks.jsonl`
- 所有 `unverified` 條目寫入 `out/debug/unverified.jsonl` 供人工檢視

**反向檢查（同樣重要）**：若 `content_block` 與來源的相似度**過高**（例如 > 0.98 的逐字複製率超過 80% 的 blocks），代表 LLM 只是在複製貼上而沒有理解與整合。

- 這同樣是失敗，記為 `degenerate_copy`
- 設計上，`經文原文` 型別**應該**是高複製率（那是引文），`白話解說` 與 `口頭延伸` **不應該**是。檢查須依 `type` 分別設定範圍。

**具名實體檢查**：`content_block` 中出現的人名、書名、數字、年代，必須在來源文字中出現。不在的一律標記 `unverified`。

### 5.5 禁止捷徑清單

以下行為在本專案中**明文禁止**。若實作者（含 AI coding agent）認為某項規定阻礙進度，正確做法是**停下來提出討論**，而不是繞過。

**關於偵測**

1. 不得以**固定間隔取樣**（每 N 秒抓一張）冒充換頁偵測。
2. 不得跳過 speaker/slide 分類，直接把所有幀當作投影片。
3. 不得只用逐字稿讓 LLM「猜」投影片邊界或投影片內容。

**關於理解**

4. 不得產生任何**無法溯源**的具體事實（人名、數字、引文、年代）。
5. 不得為了通過溯源檢查而讓 LLM 只做複製貼上（反向檢查即為此設）。
6. 不得在 S4 額度耗盡時**靜默**改用本地模型。本地 fallback 必須是明確的設定開關，且輸出須標記 `model_used`。

**關於測試**

7. **不得為了讓測試通過而調低 §5.2 的門檻**。門檻是設計目標，不是可調參數。
8. **不得縮小測試集**或移除對抗樣本（§5.1 的 A1–A7 為必選）。
9. 不得把 §5.3 的 assert 改成 warning 或 log。
10. e2e 測試**不得用 mock 取代真實模型呼叫**。單元測試可以 mock，但檔名須含 `_unit_`，且 e2e 測試須獨立存在並實際跑通至少一支影片。
11. 不得在測試中使用「跑過就好」的斷言（如 `assert result is not None`）取代量化門檻。

**關於額度**

12. 不得嘗試以任何方式將 Google AI Pro/Ultra 的**訂閱額度**接入程式化呼叫。額度來源固定為 Gemini API key。
13. 不得靠撞 429 來探測額度上限；必須主動用 quota ledger 預估與節流。

### 5.6 人工抽檢協議

自動檢查無法涵蓋「內容是否真的有用」。因此：

- 每批次跑完後，從 `out/debug/` 隨機抽 **3 個 segment** 人工檢視
- 檢查項目：投影片圖是否為該頁最完整版本、逐字稿術語是否正確、`content_block` 分類是否合理、chunk 是否自足
- 結果記入 `out/review_log.md`（日期、抽檢 segment、通過/問題、處置）
- **連續兩批出現同類問題 → 停止批次處理，回頭修 pipeline**

---

## 6. 額度與續跑機制

### 6.1 Quota Ledger

`out/quota.db`（SQLite），記錄：

```
(timestamp, model, request_count, input_tokens, output_tokens, segment_id, status)
```

**主動節流**：每次呼叫前估算 token 用量，與當日已用量相加，超過安全水位（設為配額的 90%）即停止本日處理。**不靠撞 429**——429 會讓做到一半的 segment 白費。

### 6.2 重置時間

Gemini API 的 RPD 於**太平洋時間午夜**重置。換算台灣時間：

- 夏令時間（PDT, UTC−7）→ **15:00**
- 冬令時間（PST, UTC−8）→ **16:00**

排程須依此設定，**不是台灣午夜**。實作上直接用 `zoneinfo` 以 `America/Los_Angeles` 計算，不要寫死時差。

### 6.3 續跑語意

- 每支影片的 `state.json` 記錄各階段完成狀態與**參數 hash**
- 參數 hash 變更 → 該階段及其下游需重跑；上游不動
- S4 以 segment 為粒度斷點（`07_understanding/` 每 segment 一檔）
- 重啟時掃描 `work/`，重建待辦佇列，從中斷處繼續

### 6.4 Producer / Consumer 分離

兩個獨立可執行的入口：

- `pipeline prepare <playlist>` — 跑 S0–S3，不受額度限制，可一次跑完整個系列
- `pipeline understand` — 跑 S4–S6，消化 buffer 直到額度耗盡，自動停止

這讓額度重置時工作已就緒。

### 6.5 預算估算（供驗收參考）

以 42 分鐘影片、約 30 張投影片估算：

- S4：3 segment/呼叫 → 約 10–12 次；S5：1–2 次 → **≈ 14 requests/支**
- token：約 70k/支
- Flash-Lite free tier（1,000 RPD）→ 理論上限遠高於實際需求
- **保守預期：每日 15–50 支**

若實測顯著低於此，先檢查是否 CV 前置過濾失效（候選幀過多）。

---

## 7. 分期範圍

### 7.1 Phase 0 — 骨架與驗證框架（先於任何功能程式碼）

- 專案結構、設定檔、logging
- IR schema 的 dataclass / pydantic 定義
- **合成測試影片產生器**（A1–A7）
- §5.3 的不變量檢查器
- 空的 e2e 測試（跑不通，但存在）

**完成條件**：跑 `pytest` 會失敗，但失敗訊息清楚指出缺少哪些實作。

### 7.2 Phase 1 — 本地管線（S0–S3，無 LLM）

- yt-dlp 取得、Whisper、抽幀、speaker/slide 分類、HMM 換頁、逐條動畫合併
- PaddleOCR、詞庫、術語校正
- 對齊

**完成條件**：`06_segments.json` 產出，且 §5.2 中所有不涉及 LLM 的門檻通過。此時人已經可以打開 `03_slides/` 和 segments 檢查品質。

### 7.3 Phase 2 — 理解（S4–S5）

- Gemini structured output、prompt、批次策略
- quota ledger、續跑
- 溯源檢查

**完成條件**：單支影片端到端產出 `08_video.json`，溯源通過率 ≥ 0.95。

### 7.4 Phase 3 — 批次與輸出（S6）

- `chunks.jsonl`、debug markdown
- playlist 批次、失敗處理、review log

**完成條件**：目標 playlist 完整跑完，人工抽檢通過。

### 7.5 v2（本次範圍外，但 schema 須預留）

- 系列級 concept registry（跨影片概念合併、增量更新）
- 多模態檢索
- Obsidian 輸出

**schema 預留**：`chunks.jsonl` 的 metadata 已含 `series_id` 與 `episode_index`，v2 可直接在其上建索引，不需重跑。

---

## 8. 環境

- **OS**：Ubuntu 24.04
- **GPU**：RTX 5070 Ti（16GB GDDR7, sm_120 Blackwell）
- **隔離**：conda

**兩個環境**：

| 環境 | 用途 | 內容 |
|---|---|---|
| `pipe-cpu` | S0, S1b, S3(部分), S6 | yt-dlp, opencv, numpy, pydantic |
| `pipe-gpu` | S1a, S2, S3, S4 fallback | torch, faster-whisper, paddleocr, sentence-transformers |

分開的理由：S1b（抽幀、CV）不需要 GPU，可在 GPU 忙於 Whisper 時平行處理其他影片。

**兩個已知的雷**：

1. **PyTorch 必須從官方 index 用 pip 安裝**，不要用 conda-forge 版本。Blackwell 是 sm_120，conda 版容易裝到不支援的建置，錯誤訊息是 `no kernel image is available for execution on the device`，很難查。conda 只負責 Python 版本與環境隔離。
2. **ffmpeg 用系統 apt 版本**，不要用 conda 的——conda 版編碼器支援常有缺漏，抽幀會出現難以解釋的行為。

---

## 9. 開放問題與風險

| 項目 | 風險 | 緩解 |
|---|---|---|
| 直排文字 OCR | PaddleOCR 對直排的準確度未實測 | 詞庫只求高頻術語覆蓋；精確理解交給 S4 VLM。Phase 1 完成後以真實投影片實測並記錄 |
| 文言文 ASR | Whisper 對文言文的表現未知 | 這正是術語校正存在的理由。若 raw 錯誤率過高，考慮把系列詞庫餵進 `initial_prompt` |
| Gemini 免費額度變動 | Google 曾大幅調降免費額度 | Ledger 讀取實際配額而非寫死；模型設定可替換 |
| 逐條動畫的判定 | 「後一段包含前一段」的規則可能誤判 | A2 為必測樣本；若真實影片表現不佳，退回保守策略（寧可多切，人工合併） |
| 版權 | 素材為他人著作 | 個人知識庫用途，不對外散布。輸出保留 `url` 與時間戳，便於回到原始出處 |

---

## 附錄 A：參考實作

| 專案 | 可借鑑之處 | 注意 |
|---|---|---|
| `patrickmineault/vid2slides` | ffmpeg 抽幀、滿版人臉過濾、HMM 換頁、ROI 裁切 | 2020 年程式碼，環境已不可用。**移植演算法，不 fork** |
| `HHousen/lecture2notes` | 九步驟 pipeline 架構、透視裁切、圖表抽取 | **AGPL-3.0**，只讀不抄 |
| `SliTraNet`（arXiv 2202.03540） | 換頁偵測評估方法、F1 基準值 | 用於校準 §5.2 的門檻 |
| Video-RAC | 依投影片邊界做 adaptive chunking 優於硬切的實證 | 支持本設計的 chunk 策略 |

## 附錄 B：術語

- **IR**：Intermediate Representation，本系統的中介表示（§3）
- **segment**：一張投影片 + 其對應逐字稿所構成的時間區間
- **build**：逐條動畫中的一個中間狀態
- **溯源**：content_block 可回推到 slide OCR 或逐字稿的性質
