"""量 initial_prompt 對術語準確度與 CER 的影響。

    conda run -n pipe-cpu python -m experiments.r20_lexicon.measure

人工字幕為標準答案。兩組都轉繁後再比，隔離詞庫的效果。
"""
from __future__ import annotations
import json, re
from pathlib import Path
from opencc import OpenCC

HERE = Path(__file__).parent
WORK = Path("work/zIglvjoU9vo")
CC = OpenCC("s2tw")
PAD = 2.5

TERMS = ["識蘊","五蘊","憍梵钵提","瑜伽師地論","精血凝也","形兆胚也","五行分藏",
         "六律定腑","七精開竅","宮室羅布","色受想行識","阿賴耶識","太上老君內觀經",
         "陽神為三魂","陰靈為七魄","八景神具","降真靈","元和哺食","中陰身","投胎轉世"]


def norm(s): return re.sub(r"[\s，。、！？,.!?]", "", s)


def cer(a, b):
    a, b = norm(a), norm(b)
    if not b: return 0.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1] / len(b)


def win(cues, t0, t1):
    return norm("".join(c["text"] for c in cues
                        if c["t_end"] > t0 - PAD and c["t_start"] < t1 + PAD))


def main() -> int:
    manual = json.loads((WORK / "05_transcript.json").read_text())["cues"]
    base = [{**c, "text": CC.convert(c["text"])} for c in
            json.loads((HERE.parent / "r17_whisper" / "whisper_cues.json").read_text())]
    withp = json.loads((HERE / "cues_with_prompt.json").read_text())

    print(f"{'術語':16s} {'無 prompt':>9s} {'有 prompt':>9s}")
    print("─" * 50)
    a_ok = b_ok = n = 0
    for t in TERMS:
        hits = [c for c in manual if t in c["text_raw"]]
        if not hits:
            continue
        n += 1
        cue = hits[0]
        ha = t in win(base, cue["t_start"], cue["t_end"])
        hb = t in win(withp, cue["t_start"], cue["t_end"])
        a_ok += ha; b_ok += hb
        mark = "" if ha == hb else ("   ← 修好了" if hb else "   ← 反而壞了")
        print(f"{t:16s} {'✓' if ha else '✗':>9s} {'✓' if hb else '✗':>9s}{mark}")
    print(f"\n術語正確：{a_ok}/{n} → {b_ok}/{n}")

    t0 = manual[0]["t_start"]
    mt = "".join(c["text_raw"] for c in manual)
    ca = cer("".join(c["text"] for c in base if c["t_end"] > t0), mt)
    cb = cer("".join(c["text"] for c in withp if c["t_end"] > t0), mt)
    print(f"整體字元錯誤率：{ca:.1%} → {cb:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
