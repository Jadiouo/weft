"""R41 第二部分：ngram 定候選、DP 定取捨。**不新增任何旋鈕。**

R41 第一輪量到兩者互補：
  - ngram 知道**哪裡有凹陷**（講經上位置準、STEM 上切太多）
  - DP 知道**該取幾個**（STEM 上自動選 16 刀 ≈ 真實 17，完全沒門檻）

組合方式：候選切點 = ngram 深度分數的**所有局部極大值**（不設門檻，
所以沒有 α），DP 在這個受限的集合上找最小成本路徑（所以沒有 prior）。

**兩個旋鈕都被拿掉了**，這是這個組合唯一值得做的理由——
不然就只是又一個要調的東西。
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from dp import VIDEOS, bigrams, cv, dp_segment, seg_lengths  # noqa: E402

from weft.config import Config  # noqa: E402
from weft.ir import CandidateSet, Transcript  # noqa: E402
from weft.paths import WorkPaths  # noqa: E402
from weft.stages.segment import (  # noqa: E402
    DEPTH_ALPHA, _blocks, char_ngram_vectors, enforce_min_length,
    topic_boundaries, window_similarity,
)
from weft.validation.metrics import (  # noqa: E402
    boundary_prf, boundary_string, default_window, window_diff,
)


def depth_local_maxima(scores: np.ndarray) -> list[int]:
    """深度分數的所有局部極大值。**沒有門檻**——門檻是 α，這裡不要它。"""
    d = np.zeros_like(scores)
    for i in range(len(scores)):
        left = right = scores[i]
        j = i
        while j > 0 and scores[j - 1] >= scores[j]:
            j -= 1
            left = max(left, scores[j])
        j = i
        while j < len(scores) - 1 and scores[j + 1] >= scores[j]:
            j += 1
            right = max(right, scores[j])
        d[i] = (left - scores[i]) + (right - scores[i])
    return [i for i in range(1, len(d) - 1)
            if d[i] >= d[i - 1] and d[i] >= d[i + 1] and d[i] > 0]


def dp_over_candidates(block_tokens: list[list[str]],
                       candidates: list[int]) -> list[int]:
    """只在 `candidates` 這些索引上考慮切點的最小成本路徑。

    成本與 `dp.dp_segment` 相同（Laplace 平滑的描述長度），
    差別只在**搜尋空間被 ngram 縮小了**。
    """
    n = len(block_tokens)
    if n < 2 or not candidates:
        return []
    vocab = {w for toks in block_tokens for w in toks}
    k = max(1, len(vocab))
    nodes = [0] + sorted(set(candidates)) + [n]

    best = {0: 0.0}
    back: dict[int, int] = {}
    for a_i, a in enumerate(nodes[:-1]):
        if a not in best:
            continue
        counts: Counter[str] = Counter()
        acc, total = 0.0, 0
        pos = a
        for b in nodes[a_i + 1:]:
            while pos < b:
                for w in block_tokens[pos]:
                    f = counts[w]
                    acc += (f + 1) * math.log(f + 2) - f * math.log(f + 1)
                    counts[w] = f + 1
                    total += 1
                pos += 1
            if total == 0:
                continue
            cost = -acc + total * math.log(total + k)
            if best[a] + cost < best.get(b, math.inf):
                best[b] = best[a] + cost
                back[b] = a

    cuts, at = [], n
    while at in back:
        prev = back[at]
        if prev > 0:
            cuts.append(prev)
        at = prev
    return sorted(cuts)


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
        w = WorkPaths(cfg.work_dir, vid)
        tr = Transcript.model_validate_json(w.transcript.read_text(encoding="utf-8"))
        cs = CandidateSet.model_validate_json(w.candidates.read_text(encoding="utf-8"))
        units = [c.t_start for c in tr.cues if lo <= c.t_start < hi]
        ref = boundary_string(truth, units)
        k = default_window(ref)
        ref_lens = seg_lengths(truth, lo, hi)

        blocks = _blocks(tr.cues, cfg.s3.block_chars)
        toks = [bigrams(t) for _, t in blocks]
        scores = window_similarity(
            char_ngram_vectors([t for _, t in blocks]), cfg.s3.block_window)
        cand = depth_local_maxima(scores)

        print(f"\n=== {vid}  {len(blocks)} block、真實 {len(truth)} 刀、"
              f"ngram 提供 {len(cand)} 個候選")
        print(f"{'方案':<24}{'刀數':>5}{'WindowDiff':>12}{'±10s F1':>9}"
              f"{'段長CV':>8}{'最長/最短':>10}")

        def record(label, cuts):
            inside = sorted(t for t in cuts if lo < t < hi)
            wd = window_diff(ref, boundary_string(inside, units), k)
            f1 = boundary_prf(inside, truth, 10.0).f1
            lens = seg_lengths(inside, lo, hi)
            print(f"{label:<24}{len(inside):>5}{wd:>12.3f}{f1:>9.3f}"
                  f"{cv(lens):>8.2f}{max(lens) / min(lens):>10.1f}")
            out.append({"video_id": vid, "label": label, "n": len(inside),
                        "n_truth": len(truth), "window_diff": round(wd, 3),
                        "f1_10s": round(f1, 3), "cv": round(cv(lens), 3),
                        "spread": round(max(lens) / min(lens), 1)})

        record("黃金集（參考）", truth)
        record(f"ngram α={DEPTH_ALPHA:+.2f}（現行）", enforce_min_length(
            topic_boundaries(tr.cues, cfg.s3.block_chars, cfg.s3.block_window),
            cs.duration, cfg.s3.min_segment_sec))
        record("DP 全域（prior=0）", enforce_min_length(
            [blocks[i][0] for i in dp_segment(toks)], cs.duration,
            cfg.s3.min_segment_sec))
        record("**混合：ngram 候選 + DP**", enforce_min_length(
            [blocks[i][0] for i in dp_over_candidates(toks, cand)],
            cs.duration, cfg.s3.min_segment_sec))
        nothing = window_diff(ref, "0" * len(units), k)
        print(f"{'一刀不切':<24}{0:>5}{nothing:>12.3f}{'':>9}{'':>8}{'':>10}  ← 下界")
        print(f"{'':24}參考 CV={cv(ref_lens):.2f}、"
              f"最長/最短={max(ref_lens) / min(ref_lens):.1f}")

    pathlib.Path(__file__).with_name("hybrid.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
