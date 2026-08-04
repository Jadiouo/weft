"""量測：不做 speaker/slide 分類，直接用 ink Jaccard + HMM 找靜止區段。

    python -m experiments.vlm_pipeline.measure_segments

回答三個問題：
  1. 產生幾個靜止區段？
  2. 區段長度分布？
  3. 批次 3 張一次呼叫時，需要幾次 VLM 請求？

**這是量測，不是實作。** 不加任何過濾條件——先看數字再決定。
沿用現行 pipeline 的參數（decisions.md D7／D8），只是不再先分 speaker/slide。
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from weft.config import S1bConfig  # noqa: E402
from weft.stages.detect import (  # noqa: E402
    Section,
    drop_short_sections,
    ink_jaccard,
    merge_progressive,
    viterbi_changes,
)
from weft.stages.frames import extract_frames, load_frames  # noqa: E402

WORK = REPO / "work" / "zIglvjoU9vo"


def describe(values: list[int], label: str) -> None:
    if not values:
        print(f"  {label}：無")
        return
    print(
        f"  {label}：n={len(values)} "
        f"min={min(values)}s p25={int(np.percentile(values, 25))}s "
        f"median={int(statistics.median(values))}s "
        f"p75={int(np.percentile(values, 75))}s max={max(values)}s"
    )


def main() -> int:
    p = S1bConfig()
    paths = sorted((WORK / "02_frames").glob("f_*.png"))
    if not paths:
        print(f"找不到抽出的幀：{WORK / '02_frames'}", file=sys.stderr)
        return 1

    frames = load_frames(paths, p.fps, p.downscale_short_side, p.blur_sigma,
                         p.face_min_area_ratio)
    print(f"影片：zIglvjoU9vo，{len(frames)} 幀 @ {p.fps}fps"
          f"（{len(frames) / p.fps / 60:.1f} 分鐘）\n")

    # 全片視為單一連續段落——不做 speaker/slide 分類
    distances = np.zeros(len(frames))
    for i in range(1, len(frames)):
        distances[i] = ink_jaccard(frames[i - 1].ink, frames[i].ink)

    print("ink Jaccard 分布（全片，未分類）：")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q:<3} = {np.percentile(distances[1:], q):.4f}")
    print(f"  max  = {distances[1:].max():.4f}\n")

    change_at = viterbi_changes(distances, p.hmm_self_transition, p.min_ink_change)
    starts = [0] + [i for i in change_at if i > 0]
    raw = [
        Section(start=s, end=(starts[k + 1] if k + 1 < len(starts) else len(frames)),
                build_indices=[s])
        for k, s in enumerate(starts)
    ]
    print(f"HMM 偵測：{len(raw)} 個原始區段")
    describe([s.end - s.start for s in raw], "原始區段長度")

    merged = merge_progressive(raw, frames, p.progressive_containment_ratio)
    print(f"\n視覺包含合併後：{len(merged)} 個區段"
          f"（合併掉 {len(raw) - len(merged)} 個）")
    describe([s.end - s.start for s in merged], "合併後長度")

    final = drop_short_sections(merged, frames, p.min_slide_duration_sec, p.fps)
    lengths = [s.end - s.start for s in final]
    print(f"\n濾掉短於 {p.min_slide_duration_sec:.0f}s 者後：**{len(final)} 個區段**")
    describe(lengths, "最終長度")

    for batch in (1, 3):
        print(f"\n批次 {batch} 張／次 → **{-(-len(final) // batch)} 次 VLM 請求**")

    print("\n各區段：")
    for i, s in enumerate(final):
        print(f"  {i + 1:3d} [{s.start:5d},{s.end:5d}) {s.end - s.start:4d}s"
              f"{'  build×' + str(len(s.build_indices)) if s.is_progressive else ''}")

    print(f"\n{'=' * 56}")
    print(f"判斷標準：>100 個區段須回報並暫停 → 實測 {len(final)} 個，"
          f"{'**超標**' if len(final) > 100 else '在範圍內'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
