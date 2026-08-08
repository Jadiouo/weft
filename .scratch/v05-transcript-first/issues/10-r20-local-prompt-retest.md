# 10 — R20 重測：逐段局部 prompt

**要做出什麼：** 確認「用投影片文字幫 Whisper 聽對術語」到底有沒有用 ——
用 DocWhisper 的做法，不是 R20 的做法。

**Blocked by：** 08

**Status:** ready-for-agent

R20 的結論是「解碼沒用」，但 DocWhisper 用**同一個插入點**（prompt）
拿到 WER 相對改善 14.3%。差別在實作：

| | R20 | DocWhisper |
|---|---|---|
| 粒度 | 整支影片建**一個全域 prompt** | **逐 utterance** 只餵當下那張投影片 |
| 格式 | 經文原文段落 | 詞序列 `word 1, word 2, ...` |
| 長度 | 固定 200 字 | 越長越好 |

出處見 `docs/research/2026-08-08-prior-art-transcript-first.md` §4。

**為什麼 blocked by 08**：文獻明確指出 misaligned slides **會讓結果變差**
（research §5）。分段沒修好之前，「逐段」本身就是錯的。

- [ ] 逐段局部 prompt：依 S3 對齊取當下投影片的 `slide_text`
- [ ] 詞序列格式，對照連貫文本格式
- [ ] prompt 長度掃描（驗證「越長越好」在本專案素材上成不成立）
- [ ] 對照組 = R20 的全域 200 字 + 無 prompt
- [ ] 更新 `docs/decisions.md`：R20 的結論改寫為
      「全域 200 字經文 prompt 這個做法沒用」，不是「解碼層沒用」
