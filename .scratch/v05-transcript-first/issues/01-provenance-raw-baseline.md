# 01 — 溯源基準回到 transcript_raw

**要做出什麼：** 溯源檢查說「這段話有出處」時，那個出處是**沒有被投影片
污染過**的原始逐字稿。一段內容如果只有靠校正才對得上，會被標記出來，
而不是靜默通過。

**Blocked by：** 無 — 可立即開始。

**Status:** ready-for-agent

現況：`src/weft/validation/provenance.py:325`

```python
transcript = seg.transcript_corrected or seg.transcript_raw
```

投影片文字經 S4b 校正寫進逐字稿，再拿逐字稿與投影片互相比對 —— 兩個來源
不再獨立。§5.4 已警告獨立性下降，實作比警告描述的更嚴重一級。

**不要只把變數換掉。** 那會讓通過率下降（content_block 用校正後的正確詞，
對不上 raw 的錯字）。用 R27 已寫好的 `_diagnose_wrong_source` 手法：
只在 corrected 對得上、raw 對不上的，實質溯到的是投影片。

- [ ] 溯源比對一律以 `transcript_raw` 為基準
- [ ] 只靠校正才對得上的 block，標記 `depends_on_correction`（或改判 `slide_ocr`）
- [ ] 該標記出現在 `out/debug/unverified.jsonl` 與人工複核文件裡
- [ ] 快取命中路徑也走這個邏輯（D22 的坑：只改解析路徑，續跑會沿用舊值）
- [ ] 對四支既有影片重跑，記錄修正前後的通過率與成因分布，寫進 experiments
