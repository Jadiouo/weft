"""量四組 prompt 策略對術語準確度與 CER 的影響。

    conda run -n pipe-cpu python -m experiments.r32_local_prompt.measure

人工字幕為標準答案，四組都轉繁後再比。術語清單沿用 R20 的 20 個——
**不新增也不刪減**，換清單就沒得比了。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
WORK = Path("work/zIglvjoU9vo")
PAD = 2.5

#: 沿用 R20 的清單，一字不改。
TERMS = ["識蘊", "五蘊", "憍梵钵提", "瑜伽師地論", "精血凝也", "形兆胚也", "五行分藏",
         "六律定腑", "七精開竅", "宮室羅布", "色受想行識", "阿賴耶識", "太上老君內觀經",
         "陽神為三魂", "陰靈為七魄", "八景神具", "降真靈", "元和哺食", "中陰身", "投胎轉世"]

RUNS = {
    "A 無 prompt": Path("experiments/r17_whisper/whisper_cues.json"),
    "B 全域 200 字經文": Path("experiments/r20_lexicon/cues_with_prompt.json"),
    "C 逐段詞序列": HERE / "cues_local_words.json",
    "D 逐段經文段落": HERE / "cues_local_prose.json",
}


def norm(s: str) -> str:
    return re.sub(r"[\s，。、！？,.!?：；]", "", s)


def cer(hyp: str, ref: str) -> float:
    a, b = norm(hyp), norm(ref)
    if not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1] / len(b)


#: **四組都要轉繁再比**，否則量到的是字集差異不是 prompt 的效果。
#: R17 的輸出是簡體（那一輪沒接 D24 的 s2tw），沒轉的話它的術語命中
#: 必然是 0、CER 必然虛高——那不是「無 prompt 有多差」，是我沒對齊字集。
_CC = None


def _to_traditional(text: str) -> str:
    global _CC
    if _CC is None:
        from opencc import OpenCC
        _CC = OpenCC("s2tw")
    return _CC.convert(text)


def load_rows(path: Path) -> list[tuple[float, float, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for d in data:
        if isinstance(d, dict):
            out.append((d.get("t_start", 0.0), d.get("t_end", 0.0),
                        _to_traditional(d.get("text") or d.get("text_raw") or "")))
        else:
            a, b, t = d
            out.append((a, b, _to_traditional(t)))
    return out


def reference() -> list[tuple[float, float, str]]:
    d = json.loads((WORK / "05_transcript.json").read_text(encoding="utf-8"))
    return [(c["t_start"], c["t_end"], c["text_raw"]) for c in d["cues"]]


def window_text(rows, t0: float, t1: float) -> str:
    return "".join(t for a, b, t in rows if b > t0 - PAD and a < t1 + PAD)


def main() -> int:
    ref = reference()
    ref_all = norm("".join(t for _, _, t in ref))

    print(f"{'組':<18} {'cue':>5} {'術語命中':>8} {'CER':>7}")
    results = {}
    for name, path in RUNS.items():
        if not path.exists():
            print(f"{name:<18} {'—':>5} {'（缺檔）':>8}")
            continue
        rows = load_rows(path)
        hyp_all = norm("".join(t for _, _, t in rows))
        hit = sum(1 for term in TERMS if term in hyp_all)
        results[name] = {"cues": len(rows), "terms": hit, "cer": cer(hyp_all, ref_all)}
        print(f"{name:<18} {len(rows):>5} {hit:>5}/{len(TERMS)} {results[name]['cer']:>7.3f}")

    ceiling = sum(1 for term in TERMS if term in ref_all)
    print(f"\n人工字幕本身命中 {ceiling}/{len(TERMS)} —— **這是天花板**，"
          f"字幕沒寫的詞任何 ASR 都不可能『對』")

    (HERE / "results.json").write_text(
        json.dumps({"runs": results, "ceiling": ceiling, "terms": len(TERMS)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # 逐術語看誰抓到誰沒抓到——只給總數的話看不出是哪一類詞被救回來
    print("\n逐術語（只列各組不一致的）：")
    texts = {n: norm("".join(t for _, _, t in load_rows(p)))
             for n, p in RUNS.items() if p.exists()}
    for term in TERMS:
        got = {n: (term in tx) for n, tx in texts.items()}
        if len(set(got.values())) > 1:
            marks = "  ".join(f"{n.split()[0]}{'✓' if v else '✗'}" for n, v in got.items())
            print(f"  {term:<12} {marks}   字幕{'✓' if term in ref_all else '✗'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
