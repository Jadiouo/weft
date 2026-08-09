"""R37：把 R30 的對打加上**等距對照組**，並在未飽和的容忍窗上重報。

R30 只報 ±20s。而在當時的刀數密度下，±20s 的容忍窗覆蓋了 83–97% 的
時間軸、±30s 覆蓋 100%——**那個容忍窗量不出東西**。

「不看內容、等距切同樣多刀」是這場對打從頭到尾缺的對照組。
R23 的教訓：對照組出錯或缺席時，會偽裝成「訊號不準」或「方法有效」。
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r30_segmentation"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from bakeoff import MODELS, run_video  # noqa: E402

from weft.validation.metrics import boundary_prf  # noqa: E402

VIDEOS = ("cxrqHABhWOU", "2FjApOVIbUs", "UiKi5-Arce4")
TOLERANCES = (2.0, 5.0, 10.0, 20.0, 30.0)


def golden(video_id: str) -> dict:
    root = pathlib.Path(__file__).resolve().parents[2]
    return json.loads((root / "tests/golden" / f"{video_id}.golden.json").read_text(
        encoding="utf-8"))


def uniform(lo: float, hi: float, n: int) -> list[float]:
    """不看內容的對照組：本文區間內等距切 n 刀。"""
    step = (hi - lo) / (n + 1)
    return [lo + step * (i + 1) for i in range(n)]


def coverage(lo: float, hi: float, n: int, tol: float) -> float:
    """等距切 n 刀時，容忍窗覆蓋掉多少比例的時間軸。接近 1 就是飽和。"""
    return min(1.0, 2 * tol * n / (hi - lo))


def main() -> None:
    from sentence_transformers import SentenceTransformer

    cache = {n: SentenceTransformer(n, device="cuda") for n in MODELS}
    out = []
    for vid in VIDEOS:
        g = golden(vid)
        lo, hi = g["body_start"], g["body_end"]
        truth = sorted(b["t"] for b in g["segment_boundaries"]
                       if b["status"] == "confirmed")
        row = run_video(vid, cache, tolerances=TOLERANCES)
        row["body_sec"] = hi - lo

        # `bakeoff.slide_boundaries()` 讀 `06_segments.json`，而**票 08 之後
        # 那個檔裡裝的就是 ngram 的結果**——現在的 `slide` 列是新方法冠上
        # 舊方法的名字。三支上它與 `ngram` 逐位元相同（刀數 18/30/57，
        # R30 當時是 9/26）。直接丟掉，不報。
        #
        # 舊基線**無法重建**，除非把 pipeline 倒回票 08 之前重跑。
        # 兩支講經的歷史數字在 r30 的 results.json 裡，報告直接引用那份。
        if row["results"].get("slide") == row["results"].get("ngram"):
            row["results"].pop("slide")
            row["slide_baseline"] = "不可得：06_segments.json 已被票 08 覆寫"
        for m, d in row["results"].items():
            n = d[f"±{TOLERANCES[-1]:g}s"]["n_pred"]
            base = uniform(lo, hi, n)
            for tol in TOLERANCES:
                key = f"±{tol:g}s"
                d[key]["F1_uniform"] = round(boundary_prf(base, truth, tol).f1, 3)
                d[key]["margin"] = round(d[key]["F1"] - d[key]["F1_uniform"], 3)
                d[key]["coverage"] = round(coverage(lo, hi, n, tol), 3)
        out.append(row)

    path = pathlib.Path(__file__).with_name("results.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"寫出 {path}")


if __name__ == "__main__":
    main()
