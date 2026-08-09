"""R40 第二輪：**同刀數**的正面對決，外加段長分佈比對。

第一輪的兩把尺在 STEM 上排名相反。機制查出來了：WindowDiff 的 `k` 是
「參考平均段長的一半」，UiKi5 的 k ≈ 64 秒，所以一刀偏 60 秒仍常被算成
「數量對」。**段落越長的素材，這把尺對位置越寬容**，而固定長度切法
正是靠這個拿分。

所以要在**刀數對齊**之後比位置。同時量第三件事：預測出來的段長分佈
像不像真的——使用者指出真實段長變異 4.3–7.7 倍，固定長度結構上做不到。
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sweep import (  # noqa: E402
    VIDEOS, cuts_at_alpha, depths, fixed_cuts,
)

from weft.config import Config  # noqa: E402
from weft.ir import CandidateSet, Transcript  # noqa: E402
from weft.paths import WorkPaths  # noqa: E402
from weft.stages.segment import (  # noqa: E402
    _blocks, char_ngram_vectors, enforce_min_length, window_similarity,
)
from weft.validation.metrics import (  # noqa: E402
    boundary_prf, boundary_string, default_window, window_diff,
)

FINE_ALPHAS = [round(-0.5 + 0.05 * i, 2) for i in range(71)]  # −0.50 … +3.00


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
        blocks = _blocks(tr.cues, cfg.s3.block_chars)
        scores = window_similarity(
            char_ngram_vectors([t for _, t in blocks]), cfg.s3.block_window)

        # **刀數對齊**：找出最接近 len(truth) 刀的 α
        best_alpha, best_cuts = None, None
        for a in FINE_ALPHAS:
            c = [t for t in enforce_min_length(
                cuts_at_alpha(blocks, scores, a), cs.duration,
                cfg.s3.min_segment_sec) if lo < t < hi]
            if best_cuts is None or abs(len(c) - len(truth)) < abs(len(best_cuts) - len(truth)):
                best_alpha, best_cuts = a, c

        # 固定長度：切出**恰好** len(truth) 刀
        every = (hi - lo) / (len(truth) + 1)
        fixed = fixed_cuts(lo, hi, every)[:len(truth)]

        ref_lens = seg_lengths(truth, lo, hi)
        print(f"\n=== {vid}  真實 {len(truth)} 刀、k={k}（≈{k * 2:.0f} 個 cue 的一半）")
        print(f"{'方案':<22}{'刀數':>5}{'WindowDiff':>12}{'±10s F1':>9}"
              f"{'段長CV':>8}{'最長/最短':>10}")
        for label, cuts in (
            ("黃金集（參考）", truth),
            (f"ngram α={best_alpha:+.2f}", best_cuts),
            (f"固定每 {every:.0f}s", fixed),
        ):
            hyp = boundary_string(sorted(cuts), units)
            wd = window_diff(ref, hyp, k)
            f1 = boundary_prf(sorted(cuts), truth, 10.0).f1
            lens = seg_lengths(cuts, lo, hi)
            print(f"{label:<22}{len(cuts):>5}{wd:>12.3f}{f1:>9.3f}"
                  f"{cv(lens):>8.2f}{max(lens) / min(lens):>10.1f}")
            out.append({"video_id": vid, "label": label, "n": len(cuts),
                        "window_diff": round(wd, 3), "f1_10s": round(f1, 3),
                        "cv": round(cv(lens), 3),
                        "spread": round(max(lens) / min(lens), 1)})
        print(f"{'':22}參考段長 CV={cv(ref_lens):.2f}、"
              f"最長/最短={max(ref_lens) / min(ref_lens):.1f} 倍")

    pathlib.Path(__file__).with_name("matched.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
