"""R26 附錄：跨集重現訊號能不能當投影片分類的前濾。

想法來自 v4 判準的失敗：「這些字換一集還會不會一樣」對**單張圖**的模型
不可能判斷——它一次只看得到一張。但跨集比對是免費的 CV，不需要模型。

用法：`python cross_episode.py`（需要 work/ 下有四支影片的 03_slides）
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from PIL import Image

VIDS = ["zIglvjoU9vo", "C_CFyilE-ks", "cxrqHABhWOU", "2FjApOVIbUs"]
#: 與 §4.3 同一個短邊。灰階、不模糊——D27 量過模糊會誤併橫幅標題卡。
SHORT_SIDE = 180
#: 0.44（包裝最小）與 7.6（投影片最小）之間任一值結果相同。取中間偏低側。
THRESHOLD = 6.0

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(path: pathlib.Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    w, h = im.size
    s = SHORT_SIDE / min(w, h)
    return np.asarray(im.resize((int(w * s), int(h * s))), dtype=np.float32)


def main() -> None:
    golden, reps, images = {}, {}, {}
    for v in VIDS:
        golden[v] = json.loads((ROOT / f"tests/golden/{v}.golden.json").read_text())
        reps[v] = sorted(set(golden[v]["slide_groups"].values()))
        for s in reps[v]:
            images[v, s] = load(ROOT / f"work/{v}/03_slides/{s}.png")

    rows = []
    for v in VIDS:
        others = [k for k in images if k[0] != v]
        for s in reps[v]:
            a = images[v, s]
            mae = min(float(np.abs(a - images[k]).mean())
                      for k in others if images[k].shape == a.shape)
            rows.append((mae, v, s, golden[v]["slide_classes"][s]))

    yes = sorted(m for m, _, _, lab in rows if lab)
    no = sorted(m for m, _, _, lab in rows if not lab)
    print(f"是投影片 n={len(yes)}：最小 {yes[0]:.2f}")
    print(f"不是     n={len(no)}：最小 {no[0]:.2f}  中位 {no[len(no) // 2]:.2f}")
    print(f"\n門檻 {THRESHOLD}：")
    print(f"  抓到包裝 {sum(1 for m, _, _, l in rows if not l and m < THRESHOLD)}/{len(no)}")
    print(f"  誤殺投影片 {sum(1 for m, _, _, l in rows if l and m < THRESHOLD)}/{len(yes)}")


if __name__ == "__main__":
    main()
