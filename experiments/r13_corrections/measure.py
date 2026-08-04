"""R13 校準的量測。

    conda run -n pipe-cpu python -m experiments.r13_corrections.measure

對每個候選的程式端規則，量「授權 vs 未授權」的分離度。
判準比照 D1／R12：**分離倍數低於 2 倍即判該規則不可用**，不調參數硬拉。
"""

from __future__ import annotations

from pypinyin import Style, lazy_pinyin

from .contrast_set import CASES


def pinyin_key(text: str) -> list[str]:
    """轉成不帶聲調的拼音串。非漢字（英數）原樣保留、轉小寫。

    不帶聲調是刻意的：ASR 的同音錯字經常聲調就是對的（識蘊/時運），
    但講者口誤造成的近音錯字聲調常不同（钵/波）。帶聲調會把後者判為不近。
    """
    return [p.lower() for p in lazy_pinyin(text, style=Style.NORMAL)]


def edit_ratio(a: list[str], b: list[str]) -> float:
    """1 - 正規化編輯距離。兩者皆空視為相同。"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


# ---------------------------------------------------------------------------
# 候選規則。每個回傳一個分數，**分數越低越可疑**。
# ---------------------------------------------------------------------------

def r_pinyin(f: str, t: str) -> float:
    return edit_ratio(pinyin_key(f), pinyin_key(t))


def r_len_delta(f: str, t: str) -> float:
    """負的字數增量。插入型會得到很負的分數。"""
    return -(len(t) - len(f))


def r_insertion(f: str, t: str) -> float:
    """`to` 完整包含 `from` 且更長時，回傳負的新增字數；否則 0。"""
    if f and f in t and len(t) > len(f):
        return -(len(t) - len(f))
    return 0.0


def summarise(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    s = sorted(vals)
    return f"n={len(s)} min={s[0]:.3f} 中位={s[len(s) // 2]:.3f} max={s[-1]:.3f}"


RULES = [
    ("拼音相似度", r_pinyin, "higher_is_ok"),
    ("字數增量（負）", r_len_delta, "higher_is_ok"),
    ("插入新增字數（負）", r_insertion, "higher_is_ok"),
]


def main() -> int:
    ok = [c for c in CASES if c["authorized"]]
    bad = [c for c in CASES if not c["authorized"]]
    print(f"對照組：授權 {len(ok)} 筆、未授權 {len(bad)} 筆"
          f"（其中模型實際輸出 {sum(1 for c in CASES if c['real'])} 筆）\n")

    print("=" * 78)
    print("各候選規則的整體分離度")
    print("=" * 78)
    print(f"{'規則':18s} {'授權':>30s} {'未授權':>30s} {'分離':>7s}")
    for name, fn, _ in RULES:
        g = [fn(c["from"], c["to"]) for c in ok]
        b = [fn(c["from"], c["to"]) for c in bad]
        lo_g, hi_b = min(g), max(b)
        sep = (lo_g / hi_b) if hi_b > 0 else float("inf") if lo_g > hi_b else 0.0
        print(f"{name:18s} {summarise(g):>30s} {summarise(b):>30s} {sep:6.2f}x")

    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("逐類別看：哪一類的違規被哪個規則抓到")
    print("=" * 78)
    kinds = []
    for c in CASES:
        if c["kind"] not in kinds:
            kinds.append(c["kind"])
    print(f"{'類別':10s} {'授權?':>5s} {'n':>3s} "
          f"{'拼音 min/max':>16s} {'增量 min/max':>14s} {'插入 min':>9s}")
    for k in kinds:
        sub = [c for c in CASES if c["kind"] == k]
        auth = sub[0]["authorized"]
        p = [r_pinyin(c["from"], c["to"]) for c in sub]
        d = [len(c["to"]) - len(c["from"]) for c in sub]
        i = [r_insertion(c["from"], c["to"]) for c in sub]
        print(f"{k:10s} {('授權' if auth else '違規'):>5s} {len(sub):3d} "
              f"{min(p):7.3f}/{max(p):<8.3f} {min(d):+6d}/{max(d):<+7d} {min(i):8.1f}")

    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("規則組合：先擋插入，再看剩下的拼音分離度")
    print("=" * 78)
    for thr in (1, 2, 3, 4):
        blocked_bad = [c for c in bad if -r_insertion(c["from"], c["to"]) > thr]
        blocked_ok = [c for c in ok if -r_insertion(c["from"], c["to"]) > thr]
        rest_ok = [c for c in ok if c not in blocked_ok]
        rest_bad = [c for c in bad if c not in blocked_bad]
        g = [r_pinyin(c["from"], c["to"]) for c in rest_ok]
        b = [r_pinyin(c["from"], c["to"]) for c in rest_bad]
        sep = min(g) / max(b) if b and max(b) > 0 else float("inf")
        print(f"  插入門檻 >+{thr} 字：擋掉違規 {len(blocked_bad)}/{len(bad)}、"
              f"誤殺授權 {len(blocked_ok)}/{len(ok)}；"
              f"剩下的拼音分離 {sep:.2f}x"
              f"（授權最低 {min(g):.3f} / 違規最高 {max(b):.3f}）")

    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print("若採「插入門檻 >+2 字」+「拼音下限」，逐條的判定")
    print("=" * 78)
    for cut in (0.30, 0.35, 0.40, 0.45, 0.50):
        tp = sum(1 for c in bad
                 if -r_insertion(c["from"], c["to"]) > 2 or r_pinyin(c["from"], c["to"]) < cut)
        fp = sum(1 for c in ok
                 if -r_insertion(c["from"], c["to"]) > 2 or r_pinyin(c["from"], c["to"]) < cut)
        prec_after = (len(ok) - fp) / max(1, (len(ok) - fp) + (len(bad) - tp))
        print(f"  拼音下限 {cut:.2f}：擋掉違規 {tp}/{len(bad)}、誤殺授權 {fp}/{len(ok)}"
              f"  → 放行者的 precision = {prec_after:.2f}")

    # -----------------------------------------------------------------
    # 實際採用的實作跑一遍。數字直接取自 src，報告與程式不會走鐘。
    # -----------------------------------------------------------------
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from weft.validation.corrections import unauthorized_reason  # noqa: E402

    print()
    print("=" * 78)
    print("實際採用的規則組（weft.validation.corrections）")
    print("=" * 78)
    for label, subset in (("首跑實際的 9 筆", [c for c in CASES if c["real"]]),
                          ("完整對照組", CASES)):
        kept = [c for c in subset if not unauthorized_reason(c["from"], c["to"])]
        good = sum(1 for c in kept if c["authorized"])
        print(f"  {label}：放行 {len(kept)}/{len(subset)}，"
              f"precision {good / max(1, len(kept)):.2f}")
    killed = [c for c in ok if unauthorized_reason(c["from"], c["to"])]
    print(f"  誤殺授權：{len(killed)}/{len(ok)}")
    print("  仍然放行的違規（程式端擋不到，只能靠 prompt 與 §5.6 人工抽檢）：")
    for c in bad:
        if not unauthorized_reason(c["from"], c["to"]):
            print(f"    {c['from']}→{c['to']}  ({c['kind']}，"
                  f"拼音 {r_pinyin(c['from'], c['to']):.3f})  {c['why']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
