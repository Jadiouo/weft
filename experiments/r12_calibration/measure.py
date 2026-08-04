"""R12 校準的量測。

    python -m experiments.r12_calibration.measure

比照 D1：對每個 content_type 量「忠實 vs 幻覺」的分離度。
**分離倍數低於 2 倍即判該型別不可用單一門檻**——不調參數硬拉。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from weft.validation.provenance import containment, unsupported_entities  # noqa: E402

from .contrast_set import CASES  # noqa: E402


def summarise(values: list[float]) -> str:
    if not values:
        return "n=0"
    a = np.array(values)
    return f"n={len(a)} min={a.min():.3f} 中位={np.median(a):.3f} max={a.max():.3f}"


def main() -> int:
    types = sorted({c["type"] for c in CASES})

    print("=" * 74)
    print("bigram containment（現行的正向檢查指標）")
    print("=" * 74)
    print(f"{'型別':10s} {'忠實':>28s} {'幻覺':>28s} {'分離':>7s}")

    verdicts = {}
    for t in types:
        good = [containment(c["text"], c["src"]) for c in CASES
                if c["type"] == t and c["faithful"]]
        bad = [containment(c["text"], c["src"]) for c in CASES
               if c["type"] == t and not c["faithful"]]
        sep = min(good) / max(bad) if bad and max(bad) > 0 else float("inf")
        verdicts[t] = (good, bad, sep)
        print(f"{t:10s} {summarise(good):>28s} {summarise(bad):>28s} {sep:6.2f}x")

    print()
    print("判定（門檻同 D1：分離 <2x 即不可用單一門檻）")
    for t, (good, bad, sep) in verdicts.items():
        ok = sep >= 2.0
        print(f"  {t}: {'可用' if ok else '**不可用**'}"
              f"（分離 {sep:.2f}x，忠實最低 {min(good):.3f}，幻覺最高 {max(bad):.3f}）")

    # ---------------------------------------------------------------
    print()
    print("=" * 74)
    print("具名實體檢查（§5.4 的第三道，與相似度獨立）")
    print("=" * 74)
    print("幻覺樣本中，有多少能被『具名實體不在來源中』抓到？")
    for t in types:
        bad = [c for c in CASES if c["type"] == t and not c["faithful"]]
        good = [c for c in CASES if c["type"] == t and c["faithful"]]
        caught = sum(1 for c in bad if unsupported_entities(c["text"], c["src"]))
        false_alarm = sum(1 for c in good if unsupported_entities(c["text"], c["src"]))
        print(f"  {t}: 抓到 {caught}/{len(bad)} 個幻覺；"
              f"誤報 {false_alarm}/{len(good)} 個忠實樣本")

    # ---------------------------------------------------------------
    print()
    print("=" * 74)
    print("跨語言的情形（英文投影片 → 中文描述）")
    print("=" * 74)
    from .contrast_set import SRC_SLIDE_EN

    for c in CASES:
        if c["src"] is SRC_SLIDE_EN:
            sim = containment(c["text"], c["src"])
            print(f"  {'忠實' if c['faithful'] else '幻覺'} sim={sim:.3f}  {c['text'][:46]}…")
    print("  → 忠實與幻覺**都是 0.000**。containment 在跨語言下完全沒有鑑別力。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
