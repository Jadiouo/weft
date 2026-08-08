# 03 — 門檻語意拆分：per-video 閘門 vs 全域記錄

**要做出什麼：** 看到溯源數字時，知道它在講「這支影片能不能進知識庫」
還是「整體目前跑成什麼樣」。兩者不再共用一個 0.95。

**Blocked by：** 無 — 可立即開始。

**Status:** ready-for-agent

§5.4 定義的是 per-video 閘門（單支 unverified > 5% → `needs_review`，
不進 `chunks.jsonl`）。§5.2 表格卻寫成「對象：全部」的全域驗收門檻。
R27 算的 0.838 是四支合計，而且混了四種成因。

**這不是調低門檻**（不違反 §5.5 #7），是修規格 bug。

- [ ] `PROVENANCE_PER_VIDEO_GATE = 0.95` — 真的擋 chunk 輸出
- [ ] `provenance_rate_overall` — 只記錄，不設門檻，不進 `ACCEPTANCE_THRESHOLDS`
- [ ] SDD §5.2 表格與 §5.4 改寫，兩處對這個數字的說法一致
- [ ] `test_every_acceptance_threshold_is_actually_enforced` 仍然全綠
- [ ] `docs/FROZEN.md` F2 與實作對得上
