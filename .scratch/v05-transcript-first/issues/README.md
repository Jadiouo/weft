# v0.5 收尾 — 票索引

規格：[../SPEC.md](../SPEC.md)
前人工作：[../../../docs/research/2026-08-08-prior-art-transcript-first.md](../../../docs/research/2026-08-08-prior-art-transcript-first.md)

**工作 frontier**：任何 blocker 都已完成的票。每張票在自己的 context window
做完 —— `/implement` 一張，`/clear`，再下一張。

---

## 相依圖

```
可立即開始（五張互不相干，順序隨意）
  01 溯源基準回到 raw
  02 測試不靠網路和額度
  03 門檻語意拆分
  04 S-1 比較基準
  05 CV 跨集前濾
  06 分段黃金集標註
                    │
  07 語意分段量測 ←─┘（先量再做）
       │
  08 主幹化：逐字稿驅動分段
       ├──→ 09 無投影片路徑
       ├──→ 10 R20 逐段 prompt 重測
       └──→ 14 [探索] 階層式分段
                    │
  11 跑三組素材 ←── 04, 05, 08, 09
       ├──→ 12 可用性指標
       └──→ 13 Obsidian 落地並實際查一次
```

## 清單

| # | 票 | Blocked by | 性質 |
|---|---|---|---|
| 01 | [溯源基準回到 transcript_raw](01-provenance-raw-baseline.md) | — | 修正 |
| 02 | [測試不靠網路和 API 額度](02-testing-reproducible.md) | — | 基礎建設 |
| 03 | [門檻語意拆分](03-threshold-semantics.md) | — | 規格修正 |
| 04 | [S-1 比較基準改為系列彙總](04-survey-baseline.md) | — | 規格修正 |
| 05 | [CV 跨集前濾](05-cross-episode-prefilter.md) | — | 功能 |
| 06 | [分段黃金集標註](06-segmentation-golden.md) | — | 標註 |
| 07 | [語意分段方案量測](07-segmentation-bakeoff.md) | 06 | **量測，產出決策** |
| 08 | [主幹化：逐字稿驅動分段](08-transcript-first-backbone.md) | 07 | **核心改動** |
| 09 | [無投影片路徑](09-no-slides-path.md) | 08 | 功能 |
| 10 | [R20 逐段 prompt 重測](10-r20-local-prompt-retest.md) | 08 | 量測 |
| 11 | [跑三組素材](11-run-three-corpora.md) | 04,05,08,09 | **收尾兌現** |
| 12 | [可用性指標](12-usability-metric.md) | 11 | 量測 |
| 13 | [Obsidian 落地](13-obsidian-landing.md) | 11 | **收尾兌現** |
| 14 | [[探索] 階層式分段](14-treeseg-exploration.md) | 08 | **不在收尾範圍**（FROZEN F5）|
| 15 | [`series_id` 沒流到 chunk](15-series-id-propagation.md) | — | 票 13 回報的後續，**不在收尾範圍** |
| 16 | [STEM 分段仍輸給不分段](16-stem-segmentation.md) | 需更多標註 | R40／R41 的後續，**不在收尾範圍** |

## 狀態（2026-08-09）

| # | 狀態 | 出處 |
|---|---|---|
| 01–06 | 完成 | R28、票 02、R29、票 04、票 05、票 06 |
| 07 | 完成，**結論已被 R37 修正** | R30 → `experiments/r37_segmentation_tolerance/` |
| 08 | 完成 | `src/weft/stages/segment.py` |
| 09 | 完成 | R35（`cvb4Bl93lzQ`，溯源 0.978）|
| 10 | 完成，**推翻 R20** | R32 |
| 11 | **抽樣完成**（8 支，非全跑）| R35／R36，依使用者指示「抽一些出來做就好」|
| 12 | 完成，**對照組修正後 +0.188** | R34 |
| 13 | **weft 這側完成**；vault 側需使用者在場 | R38 |
| 14／15 | 未開始，已凍結／已列後續 | FROZEN F5、票 15 |

### 收尾三條件

| 條件 | |
|---|---|
| 跑完三組素材 | ✅ 抽樣 8 支：中醫講經 ×4、無投影片 ×1、STEM ×3 |
| 產出真的被使用 | ⚠️ `chunks.jsonl` 166 筆、契約已釘住；**vault 側的 ingest 與實際查詢需使用者在場** |
| 未解的東西明確凍結 | ✅ FROZEN F11（容忍窗）、F12（chunk.id）；known-risks R29–R32 |

### 仍卡在使用者的兩件事

1. **per-video 溯源閘門的取捨**（known-risks R23）——8 支裡 4 支被擋，
   被擋掉的內容 R39 已拆解過，但「寧可漏放還是錯放」是使用者的決定。
2. `Best Partners TV` 的 URL（票 09 的第二支目標素材）。

## 建議起手

**01 或 02。** 兩張都獨立、都小、都立刻有價值：01 補掉一個真實的正確性漏洞，
02 讓後面每一張票都能被快速驗證。

06 也可以並行開始 —— 它是標註體力活，不需要寫程式，而且是 07/08 的唯一 blocker。

## 這批票的兩條紀律

1. **07 在 08 之前，不可跳過。** 提案 §6 的三個方向到現在一個都沒量測過。
   直接實作 = 把猜測寫成規格，這個 repo 已經為此付過三次代價。
2. **保留集是硬性的。** 任何在調校集上很漂亮的數字，在保留集上驗過才算數
   （R26 的教訓：v3 的 1.000 是素材的統計巧合）。
