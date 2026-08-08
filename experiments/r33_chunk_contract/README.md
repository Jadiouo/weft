# R33（票 13，weft 這側）：`chunks.jsonl` 的欄位夠不夠支撐 vault 的轉換

2026-08-08。**轉換腳本不屬於本 repo**（FROZEN F7、SDD §1.2）；
本票在 weft 這側只回答一件事：**欄位夠不夠**。

樣本：票 08 之後產出的 69 個 chunk（cxrqHABhWOU 23、2FjApOVIbUs 46）。

---

## 1. 夠的部分

票裡要的「來源可回溯性（影片、秒數、連結）」**全部具備**：

| 欄位 | 實例 |
|---|---|
| `video_id` | `cxrqHABhWOU` |
| `video_title` | 古典醫學之人體設計系列-27胰腺 |
| `t_start` / `t_end` | 0.0 / 106.339 |
| `url` | `https://www.youtube.com/watch?v=cxrqHABhWOU&t=292s` ← **帶時間戳的深連結** |
| `content_type` | 口頭延伸／白話解說／圖表描述／經文原文 |
| `provenance_kind` | transcript／slide_ocr |
| `slide_ref` | 6/6 個 `圖表描述` 都有 |

chunk 文字中位長度 57 字、最長 269 字——落在 `.raw/` 收件匣期待的範圍。

## 2. 三個缺口

### 2.1 `series_id` 與 `episode_index` 全部是 `null`

69/69 都是 null。原因是這四支是**逐支抓的**，不是從 playlist 展開的，
而 §3.5 寫的是「playlist 來源時必有」——所以現在是**正確行為**。

但票 11 的機器人學課程是 26 支的 playlist，vault 那邊很可能要靠
`episode_index` 排序。**跑 playlist 時會不會真的填上，沒有驗證過**——
現有素材測不到。列為票 11 要順帶確認的事。

### 2.2 溯源的成因在 chunk 邊界斷掉

票 01 讓 `wrong_source` 與 `depends_on_correction` 寫回 `ContentBlock`，
但**它們沒有進 `ChunkMetadata`**。

現況下這不是問題：只有通過溯源的 block 會變成 chunk，所以 `chunks.jsonl`
裡的每一筆都是 verified。但 vault 那邊看不到「這一筆的把握有多大」——
`similarity` 也沒帶過去。

**沒有現在就加**：§5.3 不變量 8 要求 metadata 全欄位完整無 null，
加欄位要同時處理舊資料，而目前**沒有任何消費端要求它**。
等 vault 側的轉換腳本真的需要時再加，那時才知道要什麼形狀。

### 2.3 `圖表描述` 有 `slide_ref` 但沒有圖檔路徑

vault 若要把投影片圖一起帶進去，得自己從 `slide_ref` 反推
`work/{video_id}/03_slides/{slide_ref}.png`——而 `work/` 不進版控，
也不保證還在。

同樣**沒有現在就加**：圖要不要進 vault 是 vault 那側的決定
（SDD §1.2 的邊界），weft 硬塞一個路徑進去反而是越界。

## 3. `terms` 只有 7/69 有值

不是缺口，是觀察：S4c 產出的 `terms` 大多是空的。
vault 的概念層若打算靠它建索引，實際拿到的東西會比預期少很多。

**沒有查原因**——那是 S4c 的 prompt 問題，不在本票範圍。記在這裡，
免得 vault 側依賴一個實際上很稀疏的欄位。

## 4. 結論

**欄位夠用。** 票 13 在 weft 這側的驗收條件（「確認 chunk metadata 足以
生成 vault 要的來源可回溯性」）**成立**。

三個缺口都**刻意不補**，理由都是同一個：現在補等於猜 vault 要什麼形狀。
`docs/FROZEN.md` F7 說得很清楚——兩個 repo 的邊界保持乾淨。

**剩下的驗收條件在 weft 這側做不到**：
「在 vault 側寫轉換腳本」「跑一次 ingest」「使用者實際查一次」
都在 `~/Documents/library`，而且最後一項需要使用者本人。
