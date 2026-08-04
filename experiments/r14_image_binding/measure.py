"""量測 S4 首跑的圖片↔區段錯位（D20）。

    conda run -n pipe-cpu python -m experiments.r14_image_binding.measure

標準答案 `labels_is_slide.json` 是**人工逐張判讀** `03_slides/*.png` 的結果，
判準與 SYSTEM_PROMPT 一致：「為了講解而製作的教材」，攝影棚佈景文字不算。

關鍵不是錯誤率本身，是**錯誤的分布**：若錯誤全部落在批次內、
沒有一個跨批次邊界，那就是綁定問題而不是模型能力問題。
"""

from __future__ import annotations

import json
from pathlib import Path

WORK = Path("work/zIglvjoU9vo")
HERE = Path(__file__).parent
BATCH = 3  # configs/default.yaml 的 understand.batch_segments


def main() -> int:
    truth = json.loads((HERE / "labels_is_slide.json").read_text())
    n_seg = len(truth)

    rows = []
    for n in range(n_seg):
        d = json.loads((WORK / "07_understanding" / f"seg_{n:03d}.json").read_text())
        slide_id = f"slide_{n + 1:03d}"
        rows.append({
            "seg": n,
            "slide": slide_id,
            "truth": truth[slide_id],
            "pred": bool(d.get("is_slide")),
            "chars": len(d.get("slide_text") or ""),
            "batch": n // BATCH,
        })

    wrong = [r for r in rows if r["truth"] != r["pred"]]
    print(f"is_slide 錯 {len(wrong)}/{n_seg} = {len(wrong) / n_seg:.1%}")
    print(f"（真值：投影片 {sum(r['truth'] for r in rows)}、"
          f"非投影片 {n_seg - sum(r['truth'] for r in rows)}）\n")

    # ------------------------------------------------------------------
    # 決定性的檢查：錯誤是否全部落在批次內？
    # 若是模型能力不足，錯誤應與批次邊界無關而隨機散布。
    # ------------------------------------------------------------------
    print("=" * 70)
    print("逐批次比對（→ 表示該批的真值序列與預測序列）")
    print("=" * 70)
    n_bad_batches = n_perm = 0
    for b in range(0, (n_seg + BATCH - 1) // BATCH):
        grp = [r for r in rows if r["batch"] == b]
        t = [r["truth"] for r in grp]
        p = [r["pred"] for r in grp]
        f = lambda xs: "".join("片" if x else "—" for x in xs)  # noqa: E731
        if t == p:
            continue
        n_bad_batches += 1
        # 這一批的預測是否只是真值的**重排**（個數相同、順序不同）？
        permuted = sorted(t) == sorted(p)
        n_perm += permuted
        print(f"  b{b:<3d} seg_{grp[0]['seg']:03d}–{grp[-1]['seg']:03d}  "
              f"真值 {f(t)} → 預測 {f(p)}"
              f"{'   ← 個數相同，只是位置換了' if permuted else ''}")

    print(f"\n出錯的批次 {n_bad_batches} 個，其中 {n_perm} 個是**批次內重排**"
          f"（投影片個數對，位置錯）")

    crossing = sum(1 for r in wrong
                   if not any(rr["batch"] == r["batch"] and rr["truth"] != rr["pred"]
                              for rr in rows if rr["seg"] != r["seg"]))
    print(f"落單的錯誤（同批次內沒有第二個錯，故無法用重排解釋）：{crossing}")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("最直接的證據：對純攝影棚鏡頭吐出投影片逐字內容")
    print("=" * 70)
    for r in rows:
        if not r["truth"] and r["pred"] and r["chars"] > 50:
            d = json.loads((WORK / "07_understanding" / f"seg_{r['seg']:03d}.json").read_text())
            print(f"  seg_{r['seg']:03d}（圖 {r['slide']} 是講者鏡頭）"
                  f"卻回了 {r['chars']} 字：{(d['slide_text'] or '')[:34]!r}…")

    print()
    print("=" * 70)
    print("反向：對真投影片回『不是投影片』")
    print("=" * 70)
    for r in rows:
        if r["truth"] and not r["pred"]:
            d = json.loads((WORK / "07_understanding" / f"seg_{r['seg']:03d}.json").read_text())
            print(f"  seg_{r['seg']:03d}（圖 {r['slide']}）：{d.get('reject_reason')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
