# 15 — `series_id` / `episode_index`：**前提是錯的，實際缺陷是回填**

**Status:** done（2026-08-13）

## 原本寫的（錯的）

> 票 04 給 `VideoProfile` 加了 `series_id`，但**它沒有流到 chunk**。
> `ir.meta.series_id` 讀的是 `VideoMeta`，那是 S0 從**單支影片**建的。
> 兩個 `series_id` 是不同的東西，名字一樣。

**查完發現這條鏈本來就是通的**：

```
resolve_targets(target)                  → [(video_id, series_id, episode_index)]
  → prepare_one(video_id, cfg, series_id, episode_index)
    → VideoMeta.series_id / episode_index
      → build_chunks → ChunkMetadata.series_id / episode_index
```

八支素材是 `None`，只因為當初**都用單支 video_id 抓的**——
那時 `None` 是**正確**的值，不是缺失。

## 實際的缺陷：回填不會發生

```python
if satisfied(Stage.S0_FETCH) and work.meta.exists():
    meta = VideoMeta.model_validate_json(...)   # 直接讀舊的
else:
    meta = local.s0_fetch(...)
    if series_id or episode_index:              # ← 只有這條路會蓋
```

一支影片**先以單支 id 抓過、之後再用 playlist 跑**時，S0 是「已滿足」的，
那段程式碼整段跳過，`series_id` 永遠補不上。

而重抓影片沒有理由（檔案就在那），所以這**不是快取失效問題，
是旁路資訊沒有回填**。

## 已做

把回填移出 `else`，兩條路都補，並且：

- **只在值真的不同時才寫檔**（無謂的寫入會讓 mtime 變，混淆事後追查）
- **不觸發重抓**（有測試釘住）

`tests/test_unit_series_metadata.py`，6 條，含兩條容易漏的：

- **單支來源時 `None` 是正確的**——沒有這條，「補上系列資訊」很容易被
  實作成從標題猜。「古典醫學之人體設計系列-**27**胰腺」裡的 27 是不是
  集數，換一個系列就不一定了（v0.2／v0.4 兩次栽在這類推廣）。
- **鏈的最後一段** `VideoMeta` → `ChunkMetadata`——中間斷掉的話，
  前面全綠而 vault 仍然拿不到。

拿掉回填會讓 3 條轉紅，驗過。

## 沒做

**沒有把既有八支的 `series_id` 補上。** 那需要用 playlist URL 重跑
`weft prepare`，會打網路。現在管線支援了，下次用 playlist 跑就會有值。
