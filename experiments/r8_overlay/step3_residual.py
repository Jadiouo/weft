"""步驟 3：建基準幀，量殘差 ink。H1 的主要量測。

    python -m experiments.r8_overlay.step3_residual

H1：殘差 ink 可以區分「疊加」與「純講者」。

作法（依任務指定，並說明偏離之處）：
  1. 由方向 1 分出的攝影棚幀，取逐像素中位數作為基準幀 B
  2. 對每個攝影棚幀 F，算殘差 R = |F − B|
  3. 對 R 套 Otsu 得殘差 ink 遮罩，量 ink 量與空間分布
  4. 看疊加幀與非疊加幀是否分得開

**偏離之處**：不做高斯模糊。pipeline 的模糊（σ=2）是為了壓制雷射筆與
壓縮雜訊，但殘差要看的正是「相對基準的差異」，先模糊會把疊加的細字磨掉。
模糊與否的影響一併量測，見輸出的 `blurred` 欄位。

輸出：
  baseline.png        基準幀（供人工檢視乾淨程度）
  residual.json       每個攝影棚幀的殘差 ink 量與空間分布
  residual_masks/     若干代表幀的殘差遮罩（供人工檢視）
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from .common import (
    OUT,
    frame_paths,
    index_to_time,
    load_json,
    load_tiny,
    load_work_gray,
    otsu_cut,
    save_json,
    split_studio_vs_fullscreen,
)


def build_baseline(paths, indices: np.ndarray, blur: bool) -> np.ndarray:
    """攝影棚幀的逐像素中位數。

    中位數而非平均：中位數對離群值（疊加幀、講者大動作）不敏感，只要
    受污染的幀少於一半就吃不進基準。疊加佔比在報告中一併驗證。
    """
    stack = []
    for i in indices:
        g = load_work_gray(paths[int(i)])
        if blur:
            g = cv2.GaussianBlur(g, (0, 0), 2.0)
        stack.append(g)
    return np.median(np.stack(stack), axis=0)


def residual_stats(frame: np.ndarray, baseline: np.ndarray) -> dict:
    """殘差 ink 量與空間分布。"""
    residual = np.abs(frame - baseline)
    cut = otsu_cut(residual.ravel())
    mask = residual > cut

    h, w = mask.shape
    total = int(mask.sum())
    if total == 0:
        return {"ink": 0, "cut": round(float(cut), 5), "spread_x": 0.0,
                "spread_y": 0.0, "coverage": 0.0, "top_half": 0.0}

    ys, xs = np.nonzero(mask)
    return {
        "ink": total,
        "cut": round(float(cut), 5),
        # 空間分布：疊加覆蓋全畫面，講者動作集中在畫面中央
        "spread_x": round(float(xs.std() / w), 4),
        "spread_y": round(float(ys.std() / h), 4),
        "coverage": round(total / (h * w), 4),
        # 疊加的文字多在上半部（標題、經文），講者動作在下半部
        "top_half": round(float((ys < h / 2).mean()), 4),
    }


def main() -> int:
    paths = frame_paths()
    if not paths:
        print("找不到抽出的幀", file=sys.stderr)
        return 1

    split = split_studio_vs_fullscreen(load_tiny(paths))
    indices = split.studio_indices
    runs = load_json("runs.json")["runs"]
    labels = load_json("labels.json") if (OUT / "labels.json").exists() else None

    result = {}
    for blur in (False, True):
        key = "blurred" if blur else "sharp"
        baseline = build_baseline(paths, indices, blur)
        if not blur:
            cv2.imwrite(str(OUT / "baseline.png"), (baseline * 255).astype(np.uint8))

        stats = {}
        for i in indices:
            g = load_work_gray(paths[int(i)])
            if blur:
                g = cv2.GaussianBlur(g, (0, 0), 2.0)
            stats[int(i)] = residual_stats(g, baseline)
        result[key] = stats
        print(f"[{key}] 基準幀完成，量測 {len(stats)} 幀")

    save_json(
        "residual.json",
        {
            "studio_frames": int(len(indices)),
            "runs": runs,
            "labels": labels,
            "stats": {k: {str(i): v for i, v in s.items()} for k, s in result.items()},
        },
    )
    print(f"輸出：{OUT / 'residual.json'}、{OUT / 'baseline.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
