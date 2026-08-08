# 07 — 語意分段方案量測（先量再做）

**要做出什麼：** 知道哪一種分段方法在**中文 ASR 逐字稿**上真的有效，
而不是照文獻直接實作。

**Blocked by：** 06

**Status:** ready-for-agent

提案 §6 的三個方向到現在**一個都沒量測過**。這張票就是量。
文獻背景見 `docs/research/2026-08-08-prior-art-transcript-first.md` §3。

候選：

1. **現行**：投影片切換驅動（baseline，已知 35% 假邊界）
2. **TextTiling + Sentence-BERT**：語意連貫度下降（Solbiati et al. 2021）
3. **方向 2**：投影片切換降級為候選邊界，需語意佐證才採納
4. **中文 subword n-gram 輔助**：對 ASR 同音錯字有韌性
   （Multi-Scale TextTiling for Chinese Broadcast News；與 R18 的 bigram 觀察一致）

**中文適用性不能假設。** TextTiling 原為英文設計，中文沒有詞邊界。

- [ ] 四種方案在黃金集上的邊界 F1（容忍窗口比照 §5.2 的 ±2 秒）
- [ ] **同時在保留集上跑** —— 只報調校集分數的結果不予採信
- [ ] 至少一支**無投影片**素材納入評測（方案 1、3 在該素材上不適用，如實記錄）
- [ ] sentence-transformers 選型：中文模型對照，純本地可跑
- [ ] 寫成 `experiments/r28_segmentation/REPORT.md`，含「哪個方案在什麼條件下贏」
- [ ] 產出對 SPEC D-B 的確認或修正 —— 若方向 2 輸了，據實改規格
