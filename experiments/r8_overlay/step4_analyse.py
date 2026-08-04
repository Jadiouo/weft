"""步驟 4：分析殘差是否能區分交叉淡化與純講者，並反查漏標的疊加。

    python -m experiments.r8_overlay.step4_analyse

輸出分離倍數、自動門檻可行性，以及長區段內部的可疑幀（供人工反查）。
"""

from __future__ import annotations

import sys

import numpy as np

from .common import OUT, load_json, otsu_cut, save_json


def summarise(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": round(float(values.min()), 1),
        "p25": round(float(np.percentile(values, 25)), 1),
        "median": round(float(np.median(values)), 1),
        "p75": round(float(np.percentile(values, 75)), 1),
        "p95": round(float(np.percentile(values, 95)), 1),
        "max": round(float(values.max()), 1),
    }


def main() -> int:
    data = load_json("residual.json")
    labels = load_json("labels.json")

    crossfade = {c["index"] for c in labels["crossfade_frames"]}

    # 純講者 = **所有非轉場的攝影棚幀**。
    #
    # 初版用「距全螢幕邊界 ≥5 幀」定義，那會把攝影機變焦造成高殘差的
    # idx 556（ink 15513）排除在外，分離倍數因此從 1.19 被美化成 2.03。
    # 那是定義造成的假分離：556 是不折不扣的純講者幀，任何實際的偵測器
    # 都會遇到它。
    import numpy as _np

    from .common import frame_paths, load_tiny, split_studio_vs_fullscreen

    split = split_studio_vs_fullscreen(load_tiny(frame_paths()))
    speaker = {int(i) for i in _np.flatnonzero(~split.is_fullscreen)} - crossfade

    report: dict = {"crossfade_count": len(crossfade), "speaker_count": len(speaker)}

    for key in ("sharp", "blurred"):
        stats = {int(k): v for k, v in data["stats"][key].items()}
        cf = np.array([stats[i]["ink"] for i in sorted(crossfade) if i in stats], dtype=float)
        sp = np.array([stats[i]["ink"] for i in sorted(speaker) if i in stats], dtype=float)

        # 分離倍數：比照 decisions.md 的算法——兩群的邊界值之比
        separation = float(cf.min() / sp.max()) if sp.max() > 0 and cf.size else 0.0
        overlap = int((sp > cf.min()).sum()) if cf.size else 0

        # 自動門檻：對全體攝影棚幀的 ink 量套 Otsu，看能不能自己找到分界
        every = np.array([v["ink"] for v in stats.values()], dtype=float)
        cut = otsu_cut(every)
        flagged = {i for i, v in stats.items() if v["ink"] > cut}
        tp = len(flagged & crossfade)
        fp = len(flagged - crossfade)
        precision = tp / len(flagged) if flagged else 0.0
        recall = tp / len(crossfade) if crossfade else 0.0

        report[key] = {
            "crossfade_ink": summarise(cf),
            "speaker_ink": summarise(sp),
            "separation_ratio": round(separation, 3),
            "speaker_frames_above_crossfade_min": overlap,
            "otsu_cut": round(float(cut), 1),
            "otsu_flagged": len(flagged),
            "otsu_precision": round(precision, 3),
            "otsu_recall": round(recall, 3),
        }

        # 空間分布：疊加覆蓋全畫面，講者動作集中在中央
        for field in ("coverage", "spread_x", "spread_y", "top_half"):
            cfv = np.array([stats[i][field] for i in sorted(crossfade) if i in stats])
            spv = np.array([stats[i][field] for i in sorted(speaker) if i in stats])
            report[key][f"{field}_crossfade"] = summarise(cfv * 1000)
            report[key][f"{field}_speaker"] = summarise(spv * 1000)

    # 反查：長區段內部有沒有被漏掉的高殘差幀（可能是緩慢淡入的疊加）
    stats = {int(k): v for k, v in data["stats"]["sharp"].items()}
    suspects = []
    for run in data["runs"]:
        a, b = run["frame_start"], run["frame_end"]
        if b - a + 1 < 10:
            continue
        inside = [(i, stats[i]["ink"]) for i in range(a, b + 1) if i in stats]
        if len(inside) < 10:
            continue
        values = np.array([v for _, v in inside], dtype=float)
        # 區段內部若有疊加，會出現遠離該段中位數的離群值
        median = np.median(values)
        mad = np.median(np.abs(values - median)) or 1.0
        for i, v in inside:
            if (v - median) / mad > 8.0:
                suspects.append({"run": run["index"], "index": i,
                                 "ink": int(v), "run_median": int(median),
                                 "z": round(float((v - median) / mad), 1)})
    suspects.sort(key=lambda s: -s["z"])
    report["long_run_suspects"] = suspects[:25]
    report["long_run_suspect_count"] = len(suspects)

    save_json("analysis.json", report)

    print(f"標註：交叉淡化 {len(crossfade)} 幀，純講者 {len(speaker)} 幀\n")
    for key in ("sharp", "blurred"):
        r = report[key]
        print(f"--- {key} ---")
        print(f"  交叉淡化 ink：{r['crossfade_ink']}")
        print(f"  純講者   ink：{r['speaker_ink']}")
        print(f"  分離倍數（交叉淡化最小 / 純講者最大）= {r['separation_ratio']}")
        print(f"  純講者中超過交叉淡化最小值的幀數 = "
              f"{r['speaker_frames_above_crossfade_min']}")
        print(f"  Otsu 自動門檻 {r['otsu_cut']} → 標出 {r['otsu_flagged']} 幀，"
              f"precision={r['otsu_precision']}, recall={r['otsu_recall']}")
        print()
    print(f"長區段內部的可疑幀（z>8）：{report['long_run_suspect_count']} 個")
    for s in report["long_run_suspects"][:10]:
        print(f"  run {s['run']:2d} idx {s['index']:5d} ink={s['ink']:6d} "
              f"(區段中位 {s['run_median']}) z={s['z']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
