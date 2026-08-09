"""S3 分段——**以逐字稿為主幹**。SDD §4.6（v0.5 改寫）。

v0.4 以前 `Segment` 由投影片切換決定。那個作法的問題不是「不夠準」，
是**結構性的**：講者對著同一張圖講完兩件事時，它一刀都不會切。

票 07 量完（`experiments/r30_segmentation/REPORT.md`，保留集 ±20 秒 F1）：

| 方案 | F1 | 召回 |
|---|---|---|
| 投影片切換驅動（v0.4） | 0.439 | 0.60 |
| 切換降為候選 + 語意佐證（SPEC 原本的 D-B） | 0.471–0.529 | 0.53–0.60 |
| **純語意** | **0.667** | 0.93 |

方向 2 輸的是結構問題：過濾器只能是投影片切換的**子集**，
永遠找不到它沒找到的邊界，因此繼承它 0.50／0.60 的召回上限。

**投影片切換因此完全不參與分段**，降級為「這一段螢幕上是哪一張」的註記。

演算法是 TextTiling（Hearst 1997），表示用**字元 n-gram**：
中文沒有詞邊界，而 R18 量過 ASR 的錯誤幾乎全是等長同音替換——
bigram 只是變稀疏不是歸零。字元共現因此不需要斷詞也不需要模型。

> **上表的 ±20 秒容忍窗已知量不出東西**（R37）：在現行刀數密度下
> 它覆蓋 80–95% 的時間軸，±30s 覆蓋 100%。「不看內容、等距切同樣多刀」
> 的對照組在 ±20s 拿得到 0.400–0.615——R30 的分數有一大半是密度給的。
>
> 選 ngram 的理由**不是**「與 Sentence-BERT 打平」（那是容忍窗的假象）。
> ±10 秒下**兩種素材互有勝負**：講經保留集 ngram 0.400 輸給 MiniLM 0.476，
> STEM 保留集 ngram 對等距的優勢 +0.30 則是三支裡最大的。
> 三支素材、42 個邊界，這個差距扛不住換方法的重量——**留著是因為零依賴**。
> 見 `experiments/r37_segmentation_tolerance/REPORT.md`。
"""

from __future__ import annotations

import logging

import numpy as np

#: TextTiling 深度門檻的 `cutoff = µ + α·σ` 裡的 α。
#:
#: **Hearst 1997 的原始設定是 −0.5，而 R40 實測那個值在三支影片上
#: 全部輸給「整支影片當一段」**（WindowDiff 0.529／0.490／0.772
#: vs 一刀不切的 0.451／0.464／0.467）——原設定在做負功。
#:
#: +0.75 是**調校集 `cxrqHABhWOU` 上的最佳值**（與 +1.0 並列，取較保守者），
#: 兩個保留集都改善且沒有打回原形：
#:
#:     α       cxrq(調校)   2Fj(保留)   UiKi5(保留 STEM)
#:     −0.50     0.529       0.490        0.772
#:     +0.75     0.360       0.359        0.562
#:
#: **STEM 仍未修好**：0.562 依然輸給一刀不切的 0.467。改善了，沒解決。
#: 根本原因是 α 是**整支影片一個門檻**，無法在同一支影片裡分區調整，
#: 而 STEM 的參考段長差 7.7 倍。見 R40 §6。
#:
#: 這個值改動會讓所有 segment_id 位移 → S4c 快取全部失效（D32）。
#: 改之前先看 `experiments/r40_granularity/REPORT.md`。
DEPTH_ALPHA: float = 0.75

log = logging.getLogger(__name__)


def char_ngram_vectors(texts: list[str], n: int = 2) -> np.ndarray:
    """字元 n-gram 的計數向量。不需要斷詞，也不需要模型。"""
    vocab: dict[str, int] = {}
    grams: list[list[str]] = []
    for t in texts:
        cleaned = "".join(t.split())
        g = [cleaned[i:i + n] for i in range(max(0, len(cleaned) - n + 1))]
        grams.append(g)
        for x in g:
            vocab.setdefault(x, len(vocab))
    mat = np.zeros((len(texts), max(1, len(vocab))), dtype=np.float32)
    for r, g in enumerate(grams):
        for x in g:
            mat[r, vocab[x]] += 1.0
    return mat


def window_similarity(vectors: np.ndarray, window: int) -> np.ndarray:
    """每個 block 間隙的左右視窗餘弦相似度。`out[i]` 對應 block i 與 i+1 之間。"""
    n = len(vectors)
    if n < 2:
        return np.zeros(0)
    out = np.zeros(n - 1)
    for i in range(n - 1):
        left = vectors[max(0, i - window + 1): i + 1].mean(axis=0)
        right = vectors[i + 1: min(n, i + 1 + window)].mean(axis=0)
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        out[i] = float(left @ right / denom) if denom else 0.0
    return out


def depth_cut_indices(scores: np.ndarray, alpha: float = DEPTH_ALPHA) -> list[int]:
    """TextTiling 的深度分數與斷點選取。回傳「在 block i 與 i+1 之間切」的 i。

    門檻是文獻裡的參數化形式 `cutoff = µ + α·σ`；Hearst 1997 的原始設定
    是 α = −0.5。**仍然不把正確答案的數量餵給方法**——後者會讓任何量測
    變成「給它 N 個答案它就切 N 刀」。

    α 由 R40 在調校集上選出，見 `DEPTH_ALPHA`。
    """
    if len(scores) < 3:
        return []
    depths = np.zeros_like(scores)
    for i in range(len(scores)):
        left = scores[i]
        j = i
        while j > 0 and scores[j - 1] >= scores[j]:
            j -= 1
            left = max(left, scores[j])
        right = scores[i]
        j = i
        while j < len(scores) - 1 and scores[j + 1] >= scores[j]:
            j += 1
            right = max(right, scores[j])
        depths[i] = (left - scores[i]) + (right - scores[i])

    cutoff = depths.mean() + alpha * depths.std()
    return [i for i in range(1, len(depths) - 1)
            if depths[i] > cutoff
            and depths[i] >= depths[i - 1] and depths[i] >= depths[i + 1]]


def _blocks(cues, block_chars: int) -> list[tuple[float, str]]:
    """把 cue 併成 TextTiling 的偽句。回傳 `(起始秒, 文字)`。"""
    out: list[tuple[float, str]] = []
    buf: list[str] = []
    start: float | None = None
    for c in cues:
        if start is None:
            start = c.t_start
        buf.append(c.text_raw)
        if sum(len(x) for x in buf) >= block_chars:
            out.append((start, "".join(buf)))
            buf, start = [], None
    if buf and start is not None:
        out.append((start, "".join(buf)))
    return out


def topic_boundaries(cues, block_chars: int, window: int,
                     alpha: float = DEPTH_ALPHA) -> list[float]:
    """逐字稿的話題邊界（秒）。**不看畫面。**

    沒有逐字稿或內容太短時回傳空清單——那不是錯誤，是「這支影片沒有東西
    可以據以分段」，呼叫端會退回整片一段（§4.3 的失敗行為）。
    """
    blocks = _blocks(cues, block_chars)
    if len(blocks) < 4:
        return []
    vectors = char_ngram_vectors([t for _, t in blocks])
    scores = window_similarity(vectors, window)
    return [blocks[i + 1][0] for i in depth_cut_indices(scores, alpha)]


def enforce_min_length(boundaries: list[float], duration: float,
                       min_sec: float) -> list[float]:
    """丟掉會產生過短區段的邊界。

    **從前往後貪心**，不是取局部最佳：切點的順序有意義（後面的切點是在
    前一段已經成立的前提下才有意義），而且貪心是可預測的——同樣的輸入
    永遠給同樣的輸出，不會因為排序穩定性之類的細節而漂移。
    """
    out: list[float] = []
    last = 0.0
    for t in sorted(boundaries):
        if t - last >= min_sec and duration - t >= min_sec:
            out.append(t)
            last = t
    return out
