# 13 — 落地到 Obsidian 並實際查一次

**要做出什麼：** chunk 真的進了 vault，而且使用者用它查到了東西。
這是「跑完一個真實 playlist」的兌現。

**Blocked by：** 11

**Status:** ready-for-agent

**轉換腳本不屬於本 repo**（`docs/FROZEN.md` F7，SDD §1.2）。
它屬於 vault 的 `_meta/`，接上既有的 `.raw/` → ingest 管線。

本票在 weft 這側只負責：確認 `chunks.jsonl` 的欄位足以支撐那個轉換。

- [ ] 確認 chunk metadata 足以生成 vault 要的來源可回溯性（影片、秒數、連結）
- [ ] 在 vault 側寫轉換腳本（`_meta/`），產出 `.raw/` 收件匣格式的 markdown
- [ ] 跑一次既有的 ingest 流程，chunk 進到概念層／實體層
- [ ] **使用者實際查一次** —— 提一個真問題，看知識庫答不答得出來
- [ ] 若欄位不足，回報成 weft 這側的後續票，不要在 vault 側硬湊
