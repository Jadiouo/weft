"""評分：本地 VLM 對上 Gemini Flash Lite（S4 的第 1、2 項工作）。

    conda run -n pipe-cpu python -m experiments.r14_local_vlm.score

`is_slide` 對人工標註算準確率；`slide_text` 分兩塊看——
**自我一致性**（同一張投影片重複出現多次，各次轉錄的差異＝雜訊底線，
不需要標準答案）與**字元錯誤率**（對人工繕打的標準答案，只做少數幾張）。

Gemini 的數字取自首跑，**帶著 D20 的錯位**——所以它的 is_slide 準確率
不是「Flash Lite 的能力」而是「當時那條管線的表現」。兩者要分開讀。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
LABELS = HERE.parent / "r14_image_binding" / "labels_is_slide.json"
WORK = Path("work/zIglvjoU9vo")

#: 同一張投影片在影片中重複出現的組別——用來量自我一致性。
REPEATS = {
    "內觀經": ["slide_015", "slide_017", "slide_019", "slide_021", "slide_023",
               "slide_025", "slide_027", "slide_029", "slide_031", "slide_033"],
    "一月為胞": ["slide_034", "slide_036", "slide_038"],
}

#: 人工繕打的標準答案（逐字，保留原字形——注意投影片用的是「周」不是「週」）。
GROUND_TRUTH = {
    "slide_034": (
        "第一周 第二周 第三周 第四周\n"
        "《一月為胞，精血凝也》\n"
        "父母的陰陽合和開始時，受精卵一個月生長成為《胞》，是父精母血的凝聚物。"
        "【此時識蘊開始進入】。\n"
        "【比喻：買房子簽合約】"
    ),
    "slide_047": "問題：為何小產大多在3月以前？",
    "slide_043": "意導氣　氣成形\n先天之氣:腎氣",
    # 全片最重要的一張（出現 10 次）。逐字繕打自 slide_031 全解析度畫面。
    # 左右兩個綠色方塊是直排，依 SYSTEM_PROMPT「由右至左、由上而下」的
    # 規定，右塊在前。
    "slide_031": (
        "太上老君內觀經\n"
        "天地媾精，陰陽布化，萬物以生。承其宿業，分靈道一。父母和合，人受其生。始。\n"
        "一月為胞，　精血凝也。\n"
        "二月成胎，　形兆胚也。\n"
        "三月陽神為三魂，動而生也。\n"
        "四月陰靈為七魄，靜鎮形也。\n"
        "五月五行分藏，　以安神也。\n"
        "六月六律定腑，　用滋靈也。\n"
        "七月七精開竅，　通光明也。\n"
        "八月八景神具，　降真靈也。\n"
        "九月宮室羅布，　以定精也。\n"
        "十月氣足，　　萬象成也。\n"
        "元和哺食，時不停也。\n"
        "色漸成、功能漸齊備\n"
        "時間軸、色身漸完成"
    ),
}


def norm(s: str | None) -> str:
    return re.sub(r"\s+", "", s or "")


def cer(pred: str, ref: str) -> float:
    """字元錯誤率＝編輯距離 ÷ 標準答案長度。可能 >1。"""
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


def load_local(model: str) -> dict:
    p = HERE / f"raw_{model.replace(':', '_')}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_gemini() -> dict:
    out = {}
    for n in range(49):
        d = json.loads((WORK / "07_understanding" / f"seg_{n:03d}.json").read_text())
        out[f"slide_{n + 1:03d}"] = {
            "parsed": {"is_slide": bool(d.get("is_slide")),
                       "slide_text": d.get("slide_text") or ""},
            "json_ok": True, "sec": None}
    return out


def main() -> int:
    truth = json.loads(LABELS.read_text())
    sids = sorted(truth)
    models = ["gemma3:4b", "gemma3:12b", "qwen2.5vl:7b"]
    data = {m: load_local(m) for m in models}
    data["gemini-3.1-flash-lite（首跑，含 D20 錯位）"] = load_gemini()

    # ---------------------------------------------------------------
    print("=" * 78)
    print("1. is_slide（對人工標註 49 張）")
    print("=" * 78)
    print(f"{'模型':34s} {'準確率':>7s} {'誤報':>5s} {'漏抓':>5s} {'JSON':>5s} {'中位秒':>6s}")
    for m, rows in data.items():
        tp = fp = fn = tn = bad = 0
        secs = []
        for sid in sids:
            r = rows.get(sid) or {}
            p = r.get("parsed")
            if p is None:
                bad += 1
                continue
            if r.get("sec"):
                secs.append(r["sec"])
            pred, exp = bool(p.get("is_slide")), truth[sid]
            tp += pred and exp
            tn += (not pred) and (not exp)
            fp += pred and (not exp)
            fn += (not pred) and exp
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        med = f"{sorted(secs)[len(secs) // 2]:.1f}" if secs else "—"
        print(f"{m:34s} {acc:6.1%} {fp:5d} {fn:5d} {49 - bad:4d}/49 {med:>6s}")
    print("\n誤報＝把講者鏡頭/片頭當成投影片（會抄佈景書法進知識庫）")
    print("漏抓＝把真投影片當成鏡頭（整段教材內容遺失）")

    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("2. slide_text 自我一致性（同一張投影片重複出現，兩兩字元差異）")
    print("=" * 78)
    print("不需要標準答案。差異＝該模型自己的雜訊底線；溯源比對的是這個欄位，")
    print("同一張圖每次抄得不一樣，代表 §5.4 的基準本身在抖。\n")
    print(f"{'模型':34s} " + " ".join(f"{k:>14s}" for k in REPEATS))
    for m, rows in data.items():
        cells = []
        for _, group in REPEATS.items():
            texts = [norm((rows.get(s, {}).get("parsed") or {}).get("slide_text"))
                     for s in group]
            texts = [t for t in texts if t]
            if len(texts) < 2:
                cells.append("n/a")
                continue
            ds = [cer(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]]
            cells.append(f"{sum(ds) / len(ds):.1%}")
        print(f"{m:34s} " + " ".join(f"{c:>14s}" for c in cells))

    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("3. slide_text 字元錯誤率（對人工繕打的標準答案）")
    print("=" * 78)
    print(f"{'模型':34s} " + " ".join(f"{s:>12s}" for s in GROUND_TRUTH))
    for m, rows in data.items():
        cells = []
        for sid, ref in GROUND_TRUTH.items():
            p = (rows.get(sid) or {}).get("parsed") or {}
            cells.append(f"{cer(p.get('slide_text') or '', ref):.1%}")
        print(f"{m:34s} " + " ".join(f"{c:>12s}" for c in cells))

    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("4. 誤報的實例：把攝影棚佈景抄成投影片內容")
    print("=" * 78)
    for m, rows in data.items():
        bad = [(s, (rows[s]["parsed"] or {}).get("slide_text") or "")
               for s in sids
               if rows.get(s, {}).get("parsed")
               and rows[s]["parsed"].get("is_slide") and not truth[s]]
        long = [(s, t) for s, t in bad if len(norm(t)) > 10]
        print(f"\n{m}：誤報 {len(bad)} 張，其中 {len(long)} 張抄了 >10 字")
        for s, t in long[:2]:
            print(f"    {s}: {norm(t)[:52]}…")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
