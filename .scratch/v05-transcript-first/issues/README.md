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
| 14 | [[探索] 階層式分段](14-treeseg-exploration.md) | 08 | **不在收尾範圍** |

## 建議起手

**01 或 02。** 兩張都獨立、都小、都立刻有價值：01 補掉一個真實的正確性漏洞，
02 讓後面每一張票都能被快速驗證。

06 也可以並行開始 —— 它是標註體力活，不需要寫程式，而且是 07/08 的唯一 blocker。

## 這批票的兩條紀律

1. **07 在 08 之前，不可跳過。** 提案 §6 的三個方向到現在一個都沒量測過。
   直接實作 = 把猜測寫成規格，這個 repo 已經為此付過三次代價。
2. **保留集是硬性的。** 任何在調校集上很漂亮的數字，在保留集上驗過才算數
   （R26 的教訓：v3 的 1.000 是素材的統計巧合）。
