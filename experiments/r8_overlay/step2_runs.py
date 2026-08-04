"""步驟 2：把攝影棚幀切成「狀態不變」的區段，每段取一幀供人工判讀。

    python -m experiments.r8_overlay.step2_runs

為什麼不用等距取樣：疊加是**短暫**的。實測 t=765.5s 與 945.5s 有疊加，
但每 20 秒取樣在 760s 與 940s 看到的都是純講者——整個錯過。

為什麼用幀間差異提出轉換點：它量的是「相鄰兩幀差多少」，與本實驗要驗證
的「與基準幀差多少」是**不同的量**。用它提案不會讓結論循環。最終標籤仍
由人眼判讀，提案只是節省時間。

**已知限制**：若疊加以極緩慢的淡入淡出出現，幀間差異會偵測不到轉換點，
該段會被併進鄰段而標成單一狀態。報告中會說明此限制。

輸出：
  runs.json           區段清單與代表幀
  runs/run_NNN.png    每段的代表幀（供人工判讀）
  runs/sheet_NN.png   聯絡表
"""

from __future__ import annotations

import subprocess
import sys

import cv2
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

#: 幀間差異的分位數門檻。刻意調低（多切）——併錯的區段會讓兩種狀態混在
#: 同一段，比多切幾段嚴重得多。
TRANSITION_PERCENTILE = 97.0
MIN_RUN_SEC = 4
GRID_COLS, GRID_ROWS = 4, 3


def consecutive_diff(indices: np.ndarray, paths: list) -> np.ndarray:
    """相鄰**攝影棚幀**之間的差異。跨越全螢幕投影片的接縫不計入。"""
    diffs = np.zeros(len(indices), dtype=np.float32)
    previous_index = None
    previous = None
    for k, i in enumerate(indices):
        g = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if previous is not None and previous_index is not None and i - previous_index == 1:
            diffs[k] = float(np.mean(np.abs(small - previous)))
        previous, previous_index = small, i
    return diffs


def build_runs(indices: np.ndarray, diffs: np.ndarray) -> list[tuple[int, int]]:
    """切成區段，回傳 `[(起始 frame index, 結束 frame index 含)]`。"""
    cut = float(np.percentile(diffs[diffs > 0], TRANSITION_PERCENTILE))
    boundaries = {0}
    for k in range(1, len(indices)):
        # 攝影棚段落之間本來就有斷點（中間插了全螢幕投影片）
        if indices[k] - indices[k - 1] > 1 or diffs[k] >= cut:
            boundaries.add(k)
    marks = sorted(boundaries) + [len(indices)]

    runs: list[tuple[int, int]] = []
    for a, b in zip(marks, marks[1:]):
        start, end = int(indices[a]), int(indices[b - 1])
        # 只有在**時間上連續**時才併入前一段。跨越全螢幕投影片的空隙去併，
        # 會讓區段橫跨兩段不相干的攝影棚畫面，代表幀還可能落在投影片上。
        contiguous = bool(runs) and start - runs[-1][1] == 1
        if b - a < MIN_RUN_SEC and contiguous:
            runs[-1] = (runs[-1][0], end)
        else:
            runs.append((start, end))
    return runs


def build_sheets(representatives: list[int], out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    per = GRID_COLS * GRID_ROWS
    for n in range(0, len(representatives), per):
        group = representatives[n : n + per]
        args = ["ffmpeg", "-v", "error", "-y"]
        for i in group:
            args += ["-i", str(FRAMES_DIR / f"f_{i + 1:05d}.png")]
        chain = []
        for k, i in enumerate(group):
            chain.append(
                f"[{k}:v]scale=360:-1,drawtext=text='{index_to_time(i):.0f}s':"
                f"x=6:y=6:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6[v{k}]"
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
        args += ["-filter_complex", ";".join(chain), "-map", label,
                 str(out_dir / f"sheet_{n // per:02d}.png")]
        subprocess.run(args, capture_output=True, check=False)


def main() -> int:
    paths = frame_paths()
    if not paths:
        print(f"找不到抽出的幀：{FRAMES_DIR}", file=sys.stderr)
        return 1

    split = split_studio_vs_fullscreen(load_tiny(paths))
    indices = split.studio_indices
    diffs = consecutive_diff(indices, paths)
    runs = build_runs(indices, diffs)

    # 代表幀取區段中點，避免落在轉場過程中
    representatives = [(a + b) // 2 for a, b in runs]

    save_json(
        "runs.json",
        {
            "transition_percentile": TRANSITION_PERCENTILE,
            "min_run_sec": MIN_RUN_SEC,
            "runs": [
                {
                    "index": k,
                    "frame_start": a,
                    "frame_end": b,
                    "t_start": round(index_to_time(a) - 0.5, 1),
                    "t_end": round(index_to_time(b) + 0.5, 1),
                    "representative": representatives[k],
                    "label": "unlabelled",  # overlay | speaker
                }
                for k, (a, b) in enumerate(runs)
            ],
        },
    )
    build_sheets(representatives, OUT / "runs")

    studio_set = set(int(i) for i in indices)
    covered = sum(len([i for i in range(a, b + 1) if i in studio_set]) for a, b in runs)
    print(f"攝影棚幀 {len(indices)} 個 → {len(runs)} 個區段（覆蓋 {covered} 幀）")
    print(f"區段長度：中位 {int(np.median([b - a + 1 for a, b in runs]))}s，"
          f"範圍 {min(b - a + 1 for a, b in runs)}–{max(b - a + 1 for a, b in runs)}s")
    print(f"代表幀聯絡表：{OUT / 'runs'}（{len(representatives)} 張，"
          f"{(len(representatives) + 11) // 12} 張表）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
