# 15 — `series_id` / `episode_index` 沒有流到 chunk

**要做出什麼：** playlist 來源的 chunk，metadata 帶得出「哪個系列、第幾集」。

**Blocked by：** —

**Status:** ready-for-agent

**來源：** 票 13 的稽核（`experiments/r38_chunk_contract_audit/REPORT.md` §4）。
**不在 v0.5 收尾範圍**，是回報出來的後續票。

## 現況

`out/chunks.jsonl` 166 筆，`series_id` 與 `episode_index` **0/166 有值**。

票 04 給 `VideoProfile` 加了 `series_id`，但 chunk 讀的是
`ir.meta.series_id`（`VideoMeta`），而 `VideoMeta` 是 S0 從**單支影片**建的。
兩個 `series_id` 是不同的東西，名字一樣。

SDD §7.5 把這兩欄列為 v2 預留，`_CHUNK_NULLABLE_FIELDS` 因此允許它們是 null——
所以**現在沒有任何測試會紅**。

## 為什麼值得做

vault 側要按系列分組。標題裡有集數
（「古典醫學之人體設計系列-**27**胰腺」），但要靠字串解析，
而那正是當初把欄位獨立出來的理由。

## 要注意的

- `weft prepare` 吃單支 id 與 playlist 兩種輸入，**只有後者有這個資訊**。
  單支輸入時 null 是正確的，不要為了填滿而猜。
- 標題解析是**最後手段**，不是主要來源：「27」是集數還是別的數字，
  換一個系列就不一定了（v0.2 與 v0.4 的錯誤都是「一支影片的觀察推廣到整個系列」）。
- 補上之後要順手檢查 `_CHUNK_NULLABLE_FIELDS`：playlist 來源時它們
  **不該**還是 nullable，否則又是一個「檢查存在但不檢查有效」。

## 驗收

- [ ] playlist 來源跑完後，chunk 的 `series_id` / `episode_index` 有值
- [ ] 單支 id 來源仍為 null，且**有測試釘住這個差別**
- [ ] 有一條反例測試：playlist 來源卻是 null 時會紅
