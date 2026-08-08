# Issue tracker — 本機 markdown

這個 repo 沒有 git remote，也不打算開 GitHub Issues。issue 以檔案形式住在 repo 裡。

## 位置

```
.scratch/<feature-slug>/
├── SPEC.md                    # /to-spec 的產出
└── issues/
    ├── 01-<slug>.md           # 依相依順序編號，blocker 在前
    ├── 02-<slug>.md
    └── ...
```

## 每張票的格式

```markdown
# <NN> — <標題>

**要做出什麼：** 這張票讓什麼端到端行為能動，從使用者角度寫，
不是逐層的實作清單。

**Blocked by：** 阻擋這張票的票號，或「無 — 可立即開始」。

**Status:** ready-for-agent

- [ ] 驗收條件 1
- [ ] 驗收條件 2
```

## 工作方式

工作 **frontier**：任何 blocker 都已完成的票。純線性鏈就是由上往下。

每張票在**自己的 context window** 裡做完 —— `/implement` 一張，
`/clear`，再下一張。票是自足的，上一張的 context 可以丟掉。

## 慣例

- 票裡**不寫具體檔案路徑與程式碼片段**，它們過期得很快。
  例外：prototype 產出的、比散文更精確地編碼了某個決策的片段
  （狀態機、schema、型別），可以內嵌並註明來自 prototype。
- 完成的票在標題後標 `✅`，不刪除 —— 它是這一輪工作的紀錄。
- 不使用 triage 標籤。這是單人專案，沒有外來的 issue 需要分流；
  `/to-tickets` 產出的票依定義就是 agent-ready。
