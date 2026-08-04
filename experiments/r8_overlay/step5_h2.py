"""步驟 5：H2——現有的 D7（ink Jaccard）／D8（視覺包含）能否直接套在殘差上。

    python -m experiments.r8_overlay.step5_h2

任務原文：「H2（若 H1 成立）」。**H1 未成立**（分離倍數 1.19 < 2），所以
這一步的結果只是參考，不構成採用建議。仍然測，是因為它回答「改動是換一個
觀測量、還是加一整條新路徑」——這個答案對 SDD 的影響差很多，即使 H1 失敗，
知道機制能不能重用仍有價值。

作法：把 pipeline 的 ink 遮罩換成殘差遮罩，其餘（ink Jaccard + 2 狀態 HMM
+ 視覺包含合併）**原封不動**沿用，看在攝影棚幀序列上會偵測出什麼。
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from weft.stages.detect import ink_jaccard, viterbi_changes  # noqa: E402
from weft.stages.frames import ink_containment  # noqa: E402

from .common import (  # noqa: E402
    frame_paths,
    load_json,
    load_tiny,
    load_work_gray,
    otsu_cut,
    save_json,
    split_studio_vs_fullscreen,
)
from .step3_residual import build_baseline  # noqa: E402


def residual_mask(frame: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    residual = np.abs(frame - baseline)
    return residual > otsu_cut(residual.ravel())


def main() -> int:
    paths = frame_paths()
    split = split_studio_vs_fullscreen(load_tiny(paths))
    indices = split.studio_indices
    labels = load_json("labels.json")
    crossfade = {c["index"] for c in labels["crossfade_frames"]}

    baseline = build_baseline(paths, indices, blur=True)

    masks = {}
    for i in indices:
        g = cv2.GaussianBlur(load_work_gray(paths[int(i)]), (0, 0), 2.0)
        masks[int(i)] = residual_mask(g, baseline)

    # 只在**時間連續**的攝影棚段落內跑，與 pipeline 的 split_runs 行為一致
    blocks: list[list[int]] = []
    for i in (int(x) for x in indices):
        if blocks and i - blocks[-1][-1] == 1:
            blocks[-1].append(i)
        else:
            blocks.append([i])

    detected: list[int] = []
    for block in blocks:
        if len(block) < 2:
            continue
        distances = np.zeros(len(block))
        for k in range(1, len(block)):
            distances[k] = ink_jaccard(masks[block[k - 1]], masks[block[k]])
        for k in viterbi_changes(distances, 0.97, 0.01):
            if k > 0:
                detected.append(block[k])

    hit = sorted(set(detected) & crossfade)
    missed = sorted(crossfade - set(detected))
    false_positive = sorted(set(detected) - crossfade)

    # D8 的視覺包含：轉場幀與其鄰居之間的 containment
    containments = []
    for i in sorted(crossfade):
        if i - 1 in masks and i in masks:
            containments.append(round(ink_containment(masks[i - 1], masks[i]), 4))

    report = {
        "detected": len(detected),
        "crossfade_total": len(crossfade),
        "hit": hit,
        "missed": missed,
        "false_positive_count": len(false_positive),
        "false_positive_sample": false_positive[:20],
        "precision": round(len(hit) / len(detected), 3) if detected else 0.0,
        "recall": round(len(hit) / len(crossfade), 3) if crossfade else 0.0,
        "containment_at_crossfade": containments,
    }
    save_json("h2.json", report)

    print("H2：把 ink 遮罩換成殘差遮罩，D7／D8 的機制原封不動沿用\n")
    print(f"  偵測到變更點 {len(detected)} 個（真實轉場 {len(crossfade)} 個）")
    print(f"  命中 {len(hit)}：{hit}")
    print(f"  漏掉 {len(missed)}：{missed}")
    print(f"  誤報 {len(false_positive)} 個，前 20：{false_positive[:20]}")
    print(f"  precision={report['precision']}  recall={report['recall']}")
    print(f"\n  轉場幀與前一幀的 containment（D8）：{containments}")
    print("  （D8 的合併門檻是 0.70——高於它代表會被當成同一頁的 build 合併掉）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
