"""量 R13 對 ASR 術語錯誤的**能力上限**。

    conda run -n pipe-cpu python -m experiments.r18_whisper_impact.r13_ceiling

R13 只校準過 precision（0.56→1.00），**recall 從未量測**。
這裡問一個更基本的問題：**假設 VLM 完美地提出了正確的校正**，
程式端的三道閘（插入／拼音／大小寫）會不會放行？

放行率就是 R13 的**上限**——實際 recall 只會更低（VLM 還得先注意到）。
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from weft.validation.corrections import unauthorized_reason, pinyin_similarity

#: 由 align.py 的對齊結果人工讀出。左為 Whisper 實際寫的，右為正確寫法。
PAIRS = [
    ("包世運", "識蘊"),
    ("攝受想行事", "色受想行識"),
    ("膠房坡體", "憍梵波提"),
    ("餘邪師弟論", "瑜伽師地論"),
    ("異地論", "瑜伽師地論"),
    ("今雪寧也", "精血凝也"),
    ("行造胚也", "形兆胚也"),
    ("五行分障", "五行分藏"),
    ("六律定府", "六律定腑"),
    ("七經開竅", "七精開竅"),
    ("功事羅布", "宮室羅布"),
    ("三環", "三魂"),
    ("八經神聚", "八景神具"),
    ("緣和補", "元和哺食"),
]


def main() -> int:
    print(f"{'Whisper 寫的':14s} {'正確':12s} {'拼音':>5s} {'長度差':>5s}  R13 判定")
    print("─" * 76)
    passed = 0
    for a, b in PAIRS:
        reason = unauthorized_reason(a, b)
        sim = pinyin_similarity(a, b)
        if reason is None:
            passed += 1
            verdict = "放行"
        else:
            verdict = f"**擋掉** — {reason[:34]}"
        print(f"{a:14s} {b:12s} {sim:5.2f} {len(b)-len(a):+5d}  {verdict}")
    print(f"\nR13 的上限：{passed}/{len(PAIRS)} = {passed/len(PAIRS):.0%} 放行")
    print("（實際 recall 只會更低——VLM 還得先注意到這些錯誤並提出正確寫法）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
