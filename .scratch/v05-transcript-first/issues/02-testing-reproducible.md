# 02 — 測試不靠網路和 API 額度也能跑

**要做出什麼：** 擱置數週後回到這個 repo，一個指令就能回答
「現在還是好的嗎」，不需要下載 302MB 影片、不需要雲端額度。

**Blocked by：** 無 — 可立即開始。

**Status:** ready-for-agent

現況：`tests/test_e2e_pipeline.py` 4 個失敗，其中至少 2 個是被雲端 429
連累的（實測 2026-08-08，473 passed / 4 failed / 4 skipped，耗時 6.5 分鐘）。

- [ ] `test_e2e_offline` — 只吃快取的 IR fixture，無網路、無 ollama，CI 可重現
- [ ] `test_e2e_live` — 標 `@pytest.mark.live`，需下載 + 本地 ollama，預設不跑
- [ ] 雲端 provider 預設停用（見 `docs/FROZEN.md` F1），程式碼保留不刪
- [ ] `pytest tests/ -q` 在沒有網路的機器上全綠
- [ ] README 或 CLAUDE.md 寫明兩層測試怎麼跑
- [ ] 環境：補上 yt-dlp 的 JS runtime（deno），消除 "some formats may be missing"
