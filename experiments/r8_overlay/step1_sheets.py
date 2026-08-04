"""步驟 1：分出攝影棚幀，產生供人工標註的聯絡表。

    python -m experiments.r8_overlay.step1_sheets

輸出：
  split.json          方向 1 的分類結果（供後續步驟重用）
  sheets/grid_NN.png  攝影棚幀的等距取樣聯絡表，供人工判讀疊加／純講者
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from .common import (
    FRAMES_DIR,
    OUT,
    frame_paths,
    index_to_time,
    load_tiny,
    save_json,
    split_studio_vs_fullscreen,
)

#: 聯絡表的取樣間隔（秒）。42 分鐘、2053 個攝影棚幀，每 20 秒取一張
#: 約 100 張，分成 9 張 3×4 的表，人工看得完。
SAMPLE_EVERY_SEC = 20
GRID_COLS, GRID_ROWS = 4, 3


def build_sheets(indices: list[int]) -> None:
    sheets_dir = OUT / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for old in sheets_dir.glob("*.png"):
        old.unlink()

    per_sheet = GRID_COLS * GRID_ROWS
    for n in range(0, len(indices), per_sheet):
        group = indices[n : n + per_sheet]
        args = ["ffmpeg", "-v", "error", "-y"]
        for i in group:
            args += ["-i", str(FRAMES_DIR / f"f_{i + 1:05d}.png")]

        # 每格左上角標時間，人工回填標註時才對得起來
        chain = []
        for k, i in enumerate(group):
            chain.append(
                f"[{k}:v]scale=360:-1,"
                f"drawtext=text='{index_to_time(i):.0f}s':x=6:y=6:fontsize=26:"
                f"fontcolor=yellow:box=1:boxcolor=black@0.6[v{k}]"
            )
        rows = []
        for r in range(0, len(group), GRID_COLS):
            row = group[r : r + GRID_COLS]
            inputs = "".join(f"[v{r + j}]" for j in range(len(row)))
            chain.append(f"{inputs}hstack=inputs={len(row)}[r{r // GRID_COLS}]")
            rows.append(f"[r{r // GRID_COLS}]")
        if len(rows) > 1:
            chain.append(f"{''.join(rows)}vstack=inputs={len(rows)}[out]")
            label = "[out]"
        else:
            label = rows[0]

        sheet = sheets_dir / f"grid_{n // per_sheet:02d}.png"
        args += ["-filter_complex", ";".join(chain), "-map", label, str(sheet)]
        subprocess.run(args, capture_output=True, check=False)


def main() -> int:
    paths = frame_paths()
    if not paths:
        print(f"找不到抽出的幀：{FRAMES_DIR}", file=sys.stderr)
        return 1

    tiny = load_tiny(paths)
    split = split_studio_vs_fullscreen(tiny)
    studio = split.studio_indices

    save_json(
        "split.json",
        {
            "total_frames": len(paths),
            "otsu_cut": round(split.cut, 6),
            "fullscreen_frames": int(split.is_fullscreen.sum()),
            "studio_frames": int(len(studio)),
            "fullscreen_ratio": round(float(split.is_fullscreen.mean()), 4),
            "distance": [round(float(x), 6) for x in split.distance],
        },
    )

    print(f"總幀數 {len(paths)}")
    print(f"Otsu 門檻 {split.cut:.4f}")
    print(f"全螢幕投影片 {int(split.is_fullscreen.sum())} 幀"
          f"（{split.is_fullscreen.mean():.1%}）")
    print(f"攝影棚 {len(studio)} 幀（{1 - split.is_fullscreen.mean():.1%}）")

    sampled = [int(i) for i in studio if index_to_time(int(i)) % SAMPLE_EVERY_SEC < 1.0]
    print(f"\n供人工判讀的攝影棚取樣：{len(sampled)} 張，"
          f"每 {SAMPLE_EVERY_SEC} 秒一張")
    build_sheets(sampled)
    save_json("sampled_studio_indices.json", sampled)
    print(f"聯絡表：{OUT / 'sheets'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
