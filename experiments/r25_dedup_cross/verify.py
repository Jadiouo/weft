"""R25：用**獨立的度量**驗證 S1c 的分組。

    conda run -n pipe-cpu python -m experiments.r25_dedup_cross.verify

**為什麼不能只看 recall/precision**：`slide_groups` 的第一版是我從
`04_dedup.json`（S1c 自己的輸出）抄的——那是循環論證，
100%/100% 是必然的不是成績。SDD §5.1(B) 第 4 條明文禁止。

人工複核**只擋得到誤併**：複核者看的是演算法給的分組，
只會問「這樣分對嗎」，不會問「有沒有該分而沒分的」。
所以漏併要靠**另一個度量**去問。

受測演算法用 ink Jaccard（Otsu 前景遮罩）；這裡用原始灰階像素的
平均絕對差——一個看前景圖樣、一個看整體亮度分布。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

VIDEOS = ["zIglvjoU9vo", "C_CFyilE-ks", "cxrqHABhWOU"]
TINY = (160, 90)


def pixel_distances(video_id: str) -> tuple[list[float], list[float]]:
    """回傳 (已合併配對的像素差, 未合併配對的像素差)。"""
    dedup = json.loads(Path(f"work/{video_id}/04_dedup.json").read_text())["slides"]
    rep = {s: (info["duplicate_of"] or s) for s, info in dedup.items()}
    sids = sorted(rep)
    tiny = {}
    for sid in sids:
        gray = cv2.imread(f"work/{video_id}/03_slides/{sid}.png", cv2.IMREAD_GRAYSCALE)
        tiny[sid] = cv2.resize(gray, TINY, interpolation=cv2.INTER_AREA).astype(np.int16)

    merged, apart = [], []
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            diff = float(np.abs(tiny[a] - tiny[b]).mean()) / 255
            (merged if rep[a] == rep[b] else apart).append(diff)
    return merged, apart


def main() -> int:
    print(f"{'影片':14s} {'已合併':>6s} {'max':>8s} {'未合併':>6s} {'min':>8s} {'間隙':>7s}")
    ok = True
    for video_id in VIDEOS:
        if not Path(f"work/{video_id}/04_dedup.json").exists():
            print(f"{video_id:14s} （尚未跑過 S1c，略過）")
            continue
        merged, apart = pixel_distances(video_id)
        if not merged:
            print(f"{video_id:14s} （沒有任何合併，無從驗證）")
            continue
        hi, lo = max(merged), min(apart)
        gap = lo / hi if hi > 0 else float("inf")
        flag = "" if lo > hi else "   ← **有重疊，分組與獨立度量不一致**"
        ok &= lo > hi
        print(f"{video_id:14s} {len(merged):6d} {hi:8.4f} {len(apart):6d} {lo:8.4f} {gap:6.2f}x{flag}")

    print()
    print("判讀：兩個度量若在**每一個**配對上都同意（未合併最小 > 已合併最大），")
    print("      表示這個分割不太可能是演算法的產物。")
    print("      **這不是說像素差是個好偵測器**——間隙只有 1.07–1.55x，")
    print("      單獨拿它當偵測器很脆弱。它的用途是驗證既有的分割。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
