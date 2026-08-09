"""R40：分段粒度掃描。**主指標是 WindowDiff，不是 boundary F1。**

三件事一起量：

1. **α 掃描**——TextTiling 的門檻 `cutoff = µ + α·σ`。現行是 α = −0.5
   （Hearst 1997 的原始設定）。文獻預告過這條路會「precision 上升、
   recall 下降、淨效果更差」（Song et al. 2016 一系），
   所以這是**一次量測，不是一次修復**。

2. **固定長度切塊對照組**——每 N 秒切一刀，完全不看內容。
   NAACL 2025 Findings 說固定長度打平語意切塊；這個 repo 已經因為
   缺對照組栽過三次。如果固定 2 分鐘就打平了，整套語意分段不值得留。

3. **黃金集自己當上界**——參考對參考，WindowDiff 必為 0。
   沒有這條，整組數字沒有標尺。

指標：WindowDiff（主）、刀數比、±10s F1 與等距對照（輔，見 R37）。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from weft.config import Config  # noqa: E402
from weft.ir import CandidateSet, Transcript  # noqa: E402
from weft.paths import WorkPaths  # noqa: E402
from weft.stages.segment import (  # noqa: E402
    _blocks,
    char_ngram_vectors,
    enforce_min_length,
    window_similarity,
)
from weft.validation.metrics import (  # noqa: E402
    boundary_prf,
    boundary_string,
    default_window,
    margin_over_uniform,
    window_diff,
)

VIDEOS = ("cxrqHABhWOU", "2FjApOVIbUs", "UiKi5-Arce4")
ALPHAS = (-0.5, 0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
FIXED_SEC = (60.0, 90.0, 120.0, 150.0, 180.0)


def depths(scores: np.ndarray) -> np.ndarray:
    """TextTiling 的深度分數。與 `segment.depth_cut_indices` 同一套算法。"""
    out = np.zeros_like(scores)
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
        out[i] = (left - scores[i]) + (right - scores[i])
    return out


def cuts_at_alpha(blocks, scores, alpha: float) -> list[float]:
    d = depths(scores)
    cutoff = d.mean() + alpha * d.std()
    idx = [
        i
        for i in range(1, len(d) - 1)
        if d[i] > cutoff and d[i] >= d[i - 1] and d[i] >= d[i + 1]
    ]
    return [blocks[i + 1][0] for i in idx]


def fixed_cuts(lo: float, hi: float, every: float) -> list[float]:
    n = int((hi - lo) // every)
    return [lo + every * (i + 1) for i in range(n) if lo + every * (i + 1) < hi]


def main() -> None:
    cfg = Config.load("configs/local.yaml")
    root = pathlib.Path(__file__).resolve().parents[2]
    out_rows = []

    for vid in VIDEOS:
        g = json.loads((root / "tests/golden" / f"{vid}.golden.json").read_text(
            encoding="utf-8"))
        truth = sorted(b["t"] for b in g["segment_boundaries"]
                       if b["status"] == "confirmed")
        lo, hi = g["body_start"], g["body_end"]
        work = WorkPaths(cfg.work_dir, vid)
        tr = Transcript.model_validate_json(work.transcript.read_text(encoding="utf-8"))
        cs = CandidateSet.model_validate_json(work.candidates.read_text(encoding="utf-8"))

        # WindowDiff 的單位是逐字稿的 cue，**只取本文區間內的**
        units = [c.t_start for c in tr.cues if lo <= c.t_start < hi]
        ref = boundary_string(truth, units)
        k = default_window(ref)

        blocks = _blocks(tr.cues, cfg.s3.block_chars)
        scores = window_similarity(
            char_ngram_vectors([t for _, t in blocks]), cfg.s3.block_window)

        def record(label: str, cuts: list[float], kind: str) -> None:
            inside = sorted(t for t in cuts if lo < t < hi)
            hyp = boundary_string(inside, units)
            f1, uni, cov = margin_over_uniform(inside, truth, lo, hi, 10.0)
            out_rows.append({
                "video_id": vid,
                "holdout": g.get("holdout", False),
                "kind": kind,
                "label": label,
                "n_cuts": len(inside),
                "n_truth": len(truth),
                "ratio": round(len(inside) / len(truth), 2) if truth else None,
                "window_diff": round(window_diff(ref, hyp, k), 3),
                "window_diff_weighted": round(
                    window_diff(ref, hyp, k, weighted=True), 3),
                "f1_10s": round(f1, 3),
                "f1_10s_uniform": round(uni, 3),
                "k": k,
                "units": len(units),
            })

        record("黃金集自己", truth, "上界")
        record("一刀不切", [], "下界")
        for alpha in ALPHAS:
            cuts = enforce_min_length(
                cuts_at_alpha(blocks, scores, alpha), cs.duration,
                cfg.s3.min_segment_sec)
            record(f"α={alpha:+.2f}", cuts, "ngram")
        for every in FIXED_SEC:
            record(f"每 {every:.0f}s", fixed_cuts(lo, hi, every), "固定長度")

    path = pathlib.Path(__file__).with_name("results.json")
    path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for vid in VIDEOS:
        rows = [r for r in out_rows if r["video_id"] == vid]
        head = rows[0]
        tag = "保留集" if head["holdout"] else "調校集"
        print(f"\n=== {vid} {tag}  真實 {head['n_truth']} 個邊界、"
              f"{head['units']} 個 cue、k={head['k']}")
        print(f"{'':<14}{'刀數':>5}{'倍':>6}{'WindowDiff':>12}{'加權':>8}"
              f"{'±10s F1':>9}{'等距':>7}")
        for r in rows:
            mark = " ←" if r["kind"] in ("上界", "下界") else ""
            print(f"{r['label']:<14}{r['n_cuts']:>5}{r['ratio']:>6.1f}"
                  f"{r['window_diff']:>12.3f}{r['window_diff_weighted']:>8.3f}"
                  f"{r['f1_10s']:>9.3f}{r['f1_10s_uniform']:>7.3f}{mark}")


if __name__ == "__main__":
    main()
