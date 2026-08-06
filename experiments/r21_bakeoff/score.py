"""S4a 選型對打的評分。

    conda run -n pipe-cpu python -m experiments.r21_bakeoff.score

三個面向，對應 S4a 的三項工作：

  is_slide      對 49 張人工標註（但只評送進 VLM 的那 21 張）
  slide_text    對 11 張人工繕打的 CER
  description   **無法自動評**——R12 量到該型別溯源分離度 0.00x。
                只印出來供人工讀，不給分數。
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.r14_local_vlm.ground_truth import EXCLUDED, GROUND_TRUTH  # noqa: E402

HERE = Path(__file__).parent
LABELS = HERE.parent / "r14_image_binding" / "labels_is_slide.json"


def norm(s: str | None) -> str:
    return re.sub(r"\s+", "", s or "")


def cer(pred: str, ref: str) -> float:
    a, b = norm(pred), norm(ref)
    if not b:
        return 0.0 if not a else 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1] / len(b)


def main() -> int:
    truth = json.loads(LABELS.read_text())
    outs = {p.stem[4:].replace("_", ":", 1): json.loads(p.read_text())
            for p in sorted(HERE.glob("out_*.json"))}
    if not outs:
        print("還沒有輸出。先跑 run.py")
        return 1
    reps = sorted({s for rows in outs.values() for s in rows})

    print("=" * 74)
    print(f"1. is_slide（送進 VLM 的 {len(reps)} 張代表幀）")
    print("=" * 74)
    print(f"{'模型':16s} {'準確率':>7s} {'誤報':>5s} {'漏抓':>5s} {'JSON':>7s} {'中位秒':>7s}")
    for m, rows in outs.items():
        ok = fp = fn = bad = 0
        secs = []
        for sid in reps:
            r = rows.get(sid) or {}
            p = r.get("parsed")
            if p is None:
                bad += 1
                continue
            secs.append(r.get("sec", 0.0))
            pred, exp = bool(p.get("is_slide")), truth[sid]
            ok += pred == exp
            fp += pred and not exp
            fn += (not pred) and exp
        n = len(reps) - bad
        med = f"{sorted(secs)[len(secs) // 2]:.1f}" if secs else "—"
        print(f"{m:16s} {ok / max(1, n):6.1%} {fp:5d} {fn:5d} "
              f"{n:4d}/{len(reps)} {med:>7s}")

    print()
    print("=" * 74)
    print(f"2. slide_text 字元錯誤率（{len(GROUND_TRUTH)} 張人工繕打）")
    print("=" * 74)
    graded = [s for s in sorted(GROUND_TRUTH) if any(s in r for r in outs.values())]
    head = "".join(f"{m.split(':')[0][:9]:>11s}" for m in outs)
    print(f"{'投影片':12s} {'字數':>5s}{head}")
    totals = {m: [] for m in outs}
    for sid in graded:
        ref = GROUND_TRUTH[sid]
        cells = ""
        for m, rows in outs.items():
            p = (rows.get(sid) or {}).get("parsed")
            if p is None:
                cells += f"{'—':>11s}"
                continue
            v = cer(p.get("slide_text") or "", ref)
            totals[m].append(v)
            cells += f"{v:10.1%} "
        print(f"{sid:12s} {len(norm(ref)):5d}{cells}")
    print("─" * 74)
    line = ""
    for m in outs:
        vals = totals[m]
        line += f"{sum(vals) / len(vals):10.1%} " if vals else f"{'—':>11s}"
    print(f"{'平均':12s} {'':5s}{line}")
    line = ""
    for m in outs:
        vals = sorted(totals[m])
        line += f"{vals[len(vals) // 2]:10.1%} " if vals else f"{'—':>11s}"
    print(f"{'中位':12s} {'':5s}{line}")

    print(f"\n（排除 {len(EXCLUDED)} 張：" +
          "、".join(f"{k} {v[:12]}" for k, v in EXCLUDED.items()) + "）")

    print()
    print("=" * 74)
    print("3. description —— **不給分數**")
    print("=" * 74)
    print("R12 量到 `圖表描述` 的溯源分離度是 0.00x，這一欄沒有任何自動檢查")
    print("抓得到編造（R16 §4：12B 把兩個綠色方塊說成一綠一黃並自創概念對應）。")
    print("以下是最難那張（內觀經）的描述，請人工讀：\n")
    for m, rows in outs.items():
        p = (rows.get("slide_031") or rows.get("slide_029") or {}).get("parsed")
        if p:
            print(f"  [{m}] {(p.get('description') or '')[:150]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
