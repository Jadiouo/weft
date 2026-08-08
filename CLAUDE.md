# weft — 影片 → 可溯源知識庫 Pipeline

每個 session 動這個 repo 之前先讀這份文件，以及 `SDD.md` 的 §5（驗證）。

---

## 這個專案在做什麼

把影片（YouTube 播放清單）批次轉成**可檢索、可溯源**的結構化知識庫，
輸出 JSONL 供向量庫匯入。最終去向是使用者的 Obsidian vault
（`~/Documents/library`），但**本專案不做 Obsidian 整合**——見 SDD §1.2。

**v0.5 起主幹改為逐字稿**（投影片降為輔助）。理由與前人工作見
`docs/research/2026-08-08-prior-art-transcript-first.md`。

---

## 環境

```bash
# Python 環境是 conda，不是 .venv
~/miniconda3/envs/pipe-cpu/bin/python     # 主要
~/miniconda3/envs/pipe-gpu/bin/python     # GPU 工作

# 測試
~/miniconda3/envs/pipe-cpu/bin/python -m pytest tests/ -q
```

> `python` 這個裸指令在這台機器上不存在。用完整路徑或先 activate。

**已知環境問題**：`yt-dlp` 缺 JS runtime（需 deno），會警告
"some formats may be missing"，字幕欄位可能抓不完整。

---

## 這個專案最重要的一條紀律

> **未驗證的假設，和實測過的事實，在文件裡必須長得不一樣。**

這個 repo 的歷史上，同一個錯誤犯過三次，每次都是把假設寫成前提：

| | 從 | 推廣到 | 後果 |
|---|---|---|---|
| v0.2 | §1.3 單支影片的觀察 | 整個系列 | chunk 內容變成攝影棚背板裝飾字，**而所有機械檢查全綠** |
| v0.4 | 第 1 集的 profile | 第 14 集 | 背景換了，分界值失效 |
| R20 | 一次 `initial_prompt` 實驗 | 「解碼層沒用」 | 與 DocWhisper 的 14.3% 改善矛盾，見 research 檔 §4 |
| R26 | 調校集 1.000 | 判準成立 | 保留集揭穿：那是素材的統計巧合 |

**寫任何規格、報告、決策時**：實測過的標出處（`D<n>`、`R<n>`、實驗路徑），
沒實測的明寫「假設，未驗證」。不確定的數字不要寫進表格裡當事實。

`§5.2` 的門檻只有 3/10 真的在驗收——見 D30 與 `DANGLING_THRESHOLDS`。
**「檢查存在」與「檢查有效」是兩件事，而前者會偽裝成後者。**

---

## 文件導覽

| 檔案 | 內容 |
|---|---|
| `SDD.md` | 主規格。**§5 驗證在 §6 實作之前，動手前必讀** |
| `docs/FROZEN.md` | **刻意不做的東西** + 恢復觸發條件。做決定前先看這裡 |
| `docs/decisions.md` | D1–D31 決策記錄，含已作廢的 |
| `docs/known-risks.md` | 已知風險，`DANGLING_THRESHOLDS` 的配套紀錄 |
| `docs/research/` | 前人工作勘查（帶引用） |
| `docs/proposals/` | 提案，含未採納的 |
| `docs/worklog/` | 每日工作日誌 |
| `experiments/r*/REPORT.md` | 每個實驗的量測結果 |
| `.scratch/<feature>/` | 進行中的 spec 與 issue |

---

## 模型配置

**不使用雲端 API（Gemini）**。2026-08-08 的決定，理由與代價見 `docs/FROZEN.md`。
程式碼保留在 `stages/providers.py`，只是不啟用。

全部走本地（ollama）。代價：溯源通過率 0.986 → 0.838，`slide_text` CER 上升。
**這是已知且接受的。**

---

## Agent skills

本 repo 已配置 [mattpocock/skills](https://github.com/mattpocock/skills)。

### Issue tracker

本機 markdown，位於 `.scratch/<feature-slug>/issues/`。見 `docs/agents/issue-tracker.md`。

### Domain docs

單一 context：根目錄 `CONTEXT.md` + `docs/adr/`。見 `docs/agents/domain.md`。

### 常用流程

```
/grill-with-docs  →  /to-spec  →  /to-tickets  →  /implement（每張票 /clear 一次）
```

**不要直接開始寫規格。** 這個 repo 的繞路史全部來自「腦內自問自答，
猜測無聲升格成前提」。先 `/grill-with-docs` 把問題問出來，
`/to-spec` 只做合成、不做訪談。
