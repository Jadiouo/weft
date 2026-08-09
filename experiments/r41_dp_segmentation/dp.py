"""R41：不需要門檻的分段——Utiyama & Isahara (2001) 的動態規劃法。

**這是探索，只在 `experiments/` 裡，不碰生產碼。**

起因：R40 量到 α 是**整支影片一個門檻**，結構上無法在同一支影片裡
「這裡切細、那裡放粗」，而 STEM 的參考段長差 7.7 倍（38s–294s）。
刀數對齊後 ngram 的段長最長/最短是 34.6 倍——數量對了，長短分配錯了。

U&I 的作法把分段變成**最小成本路徑**：

    每一段用自己的字集分佈，用 Laplace 平滑估機率
        P(w | S_i) = (f_i(w) + 1) / (n_i + k)      k = 全文相異詞數
    整份文件的成本
        C(S) = Σ_i Σ_{w ∈ S_i} −log P(w | S_i)
    在所有可能的分段裡取最小。

**段數是解出來的**：切得越細，每段的分佈越尖（成本降），但 n_i 一小、
分母就被 k 主導（成本升）。兩者自然平衡，**沒有任何門檻要調**。

單位沿用 S3 的偽句（`block_chars` 字），詞彙用**字元 bigram**——
與現行 ngram 一致，而且中文文獻證實過次詞單位對 ASR 錯誤較穩健
（`docs/research/2026-08-09-prior-art-segmentation-granularity.md` §5）。

成本用增量維護：
    C(span) = −Σ_w f(w)·log(f(w)+1) + n·log(n+k)
所以延長一格是 O(該格的 token 數)，整體 O(B²·tokens)。
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from weft.config import Config  # noqa: E402
from weft.ir import CandidateSet, Transcript  # noqa: E402
from weft.paths import WorkPaths  # noqa: E402
from weft.stages.segment import (  # noqa: E402
    DEPTH_ALPHA, _blocks, enforce_min_length, topic_boundaries,
)
from weft.validation.metrics import (  # noqa: E402
    boundary_prf, boundary_string, default_window, window_diff,
)

VIDEOS = ("cxrqHABhWOU", "2FjApOVIbUs", "UiKi5-Arce4")


def bigrams(text: str) -> list[str]:
    """只取漢字的 2 字窗。標點與英數不參與——它們不承載話題。"""
    han = "".join(c for c in text if "一" <= c <= "鿿")
    return [han[i:i + 2] for i in range(len(han) - 1)]


def dp_segment(block_tokens: list[list[str]], length_prior: float = 0.0
               ) -> list[int]:
    """回傳切點的 block 索引（在 i-1 與 i 之間切）。

    `length_prior` 是每多切一段要額外付的成本（U&I 的 P(S) 那一項）。
    **預設 0**——先看純粹的描述長度自己會選幾段。
    它在這裡是為了讓「段數對不對」與「位置準不準」可以拆開看，
    **不是拿來調到好看的旋鈕**（那樣就退回門檻法了）。
    """
    n_blocks = len(block_tokens)
    if n_blocks < 2:
        return []
    vocab = {w for toks in block_tokens for w in toks}
    k = max(1, len(vocab))

    # span_cost[i][j] 太大存不下，改成邊算邊用：對每個起點 i 往右延伸
    best = [math.inf] * (n_blocks + 1)
    back = [0] * (n_blocks + 1)
    best[0] = 0.0

    for i in range(n_blocks):
        if best[i] == math.inf:
            continue
        counts: Counter[str] = Counter()
        acc = 0.0   # Σ_w f(w)·log(f(w)+1)
        total = 0   # n
        for j in range(i, n_blocks):
            for w in block_tokens[j]:
                f = counts[w]
                acc += (f + 1) * math.log(f + 2) - f * math.log(f + 1)
                counts[w] = f + 1
                total += 1
            if total == 0:
                continue
            cost = -acc + total * math.log(total + k) + length_prior
            if best[i] + cost < best[j + 1]:
                best[j + 1] = best[i] + cost
                back[j + 1] = i

    cuts, at = [], n_blocks
    while at > 0:
        prev = back[at]
        if prev > 0:
            cuts.append(prev)
        at = prev
    return sorted(cuts)


def seg_lengths(cuts: list[float], lo: float, hi: float) -> list[float]:
    edges = [lo] + sorted(cuts) + [hi]
    return [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]


def cv(xs: list[float]) -> float:
    return statistics.stdev(xs) / statistics.mean(xs) if len(xs) > 1 else 0.0


def main() -> None:
    cfg = Config.load("configs/local.yaml")
    root = pathlib.Path(__file__).resolve().parents[2]
    out = []

    for vid in VIDEOS:
        g = json.loads((root / "tests/golden" / f"{vid}.golden.json").read_text(
            encoding="utf-8"))
        truth = sorted(b["t"] for b in g["segment_boundaries"]
                       if b["status"] == "confirmed")
        lo, hi = g["body_start"], g["body_end"]
        work = WorkPaths(cfg.work_dir, vid)
        tr = Transcript.model_validate_json(work.transcript.read_text(encoding="utf-8"))
        cs = CandidateSet.model_validate_json(work.candidates.read_text(encoding="utf-8"))
        units = [c.t_start for c in tr.cues if lo <= c.t_start < hi]
        ref = boundary_string(truth, units)
        k = default_window(ref)
        ref_lens = seg_lengths(truth, lo, hi)

        blocks = _blocks(tr.cues, cfg.s3.block_chars)
        toks = [bigrams(t) for _, t in blocks]

        print(f"\n=== {vid}  {len(blocks)} 個 block、真實 {len(truth)} 刀、k={k}")
        print(f"{'方案':<22}{'刀數':>5}{'WindowDiff':>12}{'±10s F1':>9}"
              f"{'段長CV':>8}{'最長/最短':>10}")

        def record(label: str, cuts: list[float]) -> None:
            inside = sorted(t for t in cuts if lo < t < hi)
            hyp = boundary_string(inside, units)
            wd = window_diff(ref, hyp, k)
            f1 = boundary_prf(inside, truth, 10.0).f1
            lens = seg_lengths(inside, lo, hi)
            print(f"{label:<22}{len(inside):>5}{wd:>12.3f}{f1:>9.3f}"
                  f"{cv(lens):>8.2f}{max(lens) / min(lens):>10.1f}")
            out.append({"video_id": vid, "label": label, "n": len(inside),
                        "n_truth": len(truth), "window_diff": round(wd, 3),
                        "f1_10s": round(f1, 3), "cv": round(cv(lens), 3),
                        "spread": round(max(lens) / min(lens), 1)})

        record("黃金集（參考）", truth)
        nothing_wd = window_diff(ref, "0" * len(units), k)
        print(f"{'一刀不切':<22}{0:>5}{nothing_wd:>12.3f}{0.0:>9.3f}"
              f"{0.0:>8.2f}{1.0:>10.1f}   ← 免費下界")
        out.append({"video_id": vid, "label": "一刀不切", "n": 0,
                    "n_truth": len(truth), "window_diff": round(nothing_wd, 3),
                    "f1_10s": 0.0, "cv": 0.0, "spread": 1.0})

        record(f"ngram α={DEPTH_ALPHA:+.2f}（現行）", enforce_min_length(
            topic_boundaries(tr.cues, cfg.s3.block_chars, cfg.s3.block_window),
            cs.duration, cfg.s3.min_segment_sec))

        for prior in (0.0, 100.0, 200.0, 400.0, 800.0):
            idx = dp_segment(toks, length_prior=prior)
            cuts = enforce_min_length([blocks[i][0] for i in idx],
                                      cs.duration, cfg.s3.min_segment_sec)
            record(f"DP prior={prior:.0f}", cuts)

        print(f"{'':22}參考段長 CV={cv(ref_lens):.2f}、"
              f"最長/最短={max(ref_lens) / min(ref_lens):.1f} 倍")

    pathlib.Path(__file__).with_name("results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
