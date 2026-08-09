# R38（票 13）：`chunks.jsonl` 的欄位夠不夠 vault 用

2026-08-09。票 13 在 weft 這側只負責一件事：
**確認 chunk metadata 足以支撐 vault 側的轉換與來源可回溯性。**

轉換腳本本身不屬於本 repo（FROZEN F7、SDD §1.2），它在 vault 的 `_meta/`。

---

## 0. 結論先講

| | |
|---|---|
| **回溯性夠** | `video_id` + 標題 + 起訖秒數 + **帶時間戳的深連結**，四樣都有 |
| **一個真的問題：`id` 是位置編號** | 重跑後**同一個 id 可以指到不同的內容與時間範圍**，見 §2 |
| **已補 `content_sha`** | 不改 id 的形狀（那要下游先決定），但**內容變了看得出來** |
| **一個沒寫下來的強保證** | 檔案裡**每一筆都通過了溯源**，逐支核對過，見 §3 |
| **兩個欄位全空** | `series_id`／`episode_index` 目前 0/166，見 §4 |

## 1. 現況

`out/chunks.jsonl`，166 筆，四支影片（另外四支被 per-video 閘門整支擋掉）。

```json
{
  "id": "cxrqHABhWOU#000#b00",
  "text": "講者提到胰腺是正確的稱謂，而不是胰臟。…",
  "metadata": {
    "video_id": "cxrqHABhWOU",
    "video_title": "古典醫學之人體設計系列-27胰腺",
    "t_start": 0.0, "t_end": 106.339,
    "url": "https://www.youtube.com/watch?v=cxrqHABhWOU&t=0s",
    "content_type": "口頭延伸",
    "slide_ref": null,
    "terms": [],
    "provenance_kind": "transcript",
    "series_id": null, "episode_index": null
  }
}
```

欄位填充率：

| 欄位 | 有值 | 備註 |
|---|---|---|
| `content_type` | 166/166 | 口頭延伸 120、白話解說 40、圖表描述 6 |
| `provenance_kind` | 166/166 | transcript 156、slide_ocr 10 |
| `slide_ref` | 18/166 | 其餘是純語音段落，**本來就該是 null** |
| `terms` | 31/166 | 空的多半是 R26（投影片英文 → 詞庫空） |
| `series_id` | **0/166** | §4 |
| `episode_index` | **0/166** | §4 |

## 2. `id` 是位置編號——D32 會原封不動跟著進 vault

```
id = f"{seg.segment_id}#b{b:02d}"     # render.py:110
segment_id = f"{video_id}#{序號:03d}"  # 位置
```

改 `block_chars`、換分段方法、換一版 S4c prompt——`#010` 都還是 `#010`，
但它指的時間範圍與內容會整個換掉。

**這正是 D32。** 當時 `segment_id` 的位置性讓 S4c 讀到別的時間範圍的快取
（`#010` 從 72–98s 變成 564–593s），而**所有機械檢查都是綠的**。
D32 修的是**快取鍵**（加 `input_fingerprint`）；**匯出的 id 沒修**。

差別在後果：D32 是內部錯亂，這裡是**下游以 id 當筆記識別時，
重跑一次就會在同一則筆記底下換掉內容**——而 vault 是使用者的長期資產。

**做法**：不改 id 的形狀。要改成什麼要下游先決定用什麼當識別
（內容雜湊會讓「改一個字就變新筆記」，時間起點會讓「微調分段就漂移」，
兩種都有代價）。這裡只保證**變了看得出來**：

- `ChunkMetadata.content_sha` = `sha256(text)[:16]`
- §5.3 規則 8 會**核對它與 `text` 相符**——含正反兩條測試。
  沒有那個核對，它就只是一個看起來很可靠、但可以悄悄脫節的字串
  （這個 repo 對「檢查存在 ≠ 檢查有效」有三次前科）。

**給 vault 側的建議**：以 `(video_id, t_start)` 當筆記識別，
`content_sha` 當變更偵測。**不要用 `id`。**

## 3. 一個成立但沒寫下來的保證：檔案裡每一筆都溯源通過

逐支核對 `chunks.jsonl` 的筆數與 `provenance.jsonl` 的 `verified`：

| 影片 | chunk | blocks | verified | |
|---|---|---|---|---|
| `cxrqHABhWOU` | 23 | 24 | 23 | ✓ |
| `2FjApOVIbUs` | 46 | 47 | 46 | ✓ |
| `cvb4Bl93lzQ` | 44 | 45 | 44 | ✓ |
| `jgVD7IPNTs8` | 53 | 55 | 53 | ✓ |
| `zIglvjoU9vo` | 0 | 57 | 54 | 閘門擋掉，整支不在檔案裡 |
| `C_CFyilE-ks` | 0 | 35 | 29 | 同上 |
| `UiKi5-Arce4` | 0 | 95 | 88 | 同上 |
| `xBfyWwYylSA` | 0 | 89 | 80 | 同上 |

**四支通過的影片，chunk 數與 verified 數完全相等。**
兩層過濾：未通過的 block 在 S6 被排除（§5.4），未過閘門的影片整支被排除。

vault 側因此不需要自己再判斷可信度——**但這件事原本只存在於程式碼裡**，
沒有寫成契約，也沒有測試釘住。已補進 `tests/test_e2e_offline.py`。

> 代價要一起講：**被擋掉的四支裡有 251 個 verified 的 block 一起被丟掉**。
> per-video 閘門的取捨仍是 known-risks R23 的未決事項。

## 4. `series_id` / `episode_index` 全空

票 04 給 `VideoProfile` 加了 `series_id`，但**它沒有流到 chunk**。
`ir.meta.series_id` 讀的是 `VideoMeta`，那是 S0 從單支影片建的。

對 vault 有影響：「古典醫學之人體設計系列-27胰腺」這種標題裡有集數，
但要靠字串解析才拿得到，而那正是 SDD §7.5 把這兩個欄位列為預留的原因。

**沒有現在就補。** 它需要 S0 知道自己是從哪個 playlist 來的，
而目前 `weft prepare` 吃單支 id 與 playlist 兩種輸入，
只有後者有這個資訊。這是一張後續票，不是收尾範圍。

## 5. 沒做的事

- **vault 側的轉換腳本、ingest、使用者實際查一次**——票 13 的後四項。
  那三項都在 `~/Documents/library`，不是本 repo，且需要使用者在場。
- `content_sha` 只涵蓋 `text`。metadata 改了（例如 `content_type` 重判）
  雜湊不會變。**要偵測那個得另外算**，目前下游用不到，沒做。
