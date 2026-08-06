"""以時間重疊對齊人工字幕與 Whisper，抽出真實的術語錯誤對。

    conda run -n pipe-cpu python -m experiments.r18_whisper_impact.align

不臆測 Whisper「大概會寫成什麼」——直接從對齊結果讀出來。
"""
from __future__ import annotations
import json, re
from pathlib import Path
from opencc import OpenCC

HERE = Path(__file__).parent
CC = OpenCC("s2twp")

TERMS = ["識蘊","五蘊","憍梵钵提","瑜伽師地論","精血凝也","形兆胚也","五行分藏",
         "六律定腑","七精開竅","宮室羅布","色受想行識","阿賴耶識","太上老君內觀經",
         "陽神為三魂","陰靈為七魄","八景神具","降真靈","元和哺食"]


def load():
    w = json.loads((HERE.parent / "r17_whisper" / "whisper_cues.json").read_text())
    m = json.loads(Path("work/zIglvjoU9vo/05_transcript.json").read_text())["cues"]
    return w, m


def whisper_span(w, t0: float, t1: float, pad: float = 3.0) -> str:
    """取時間上與 [t0,t1] 有重疊的 Whisper 文字（左右各放寬 pad 秒）。"""
    return CC.convert("".join(
        c["text"] for c in w if c["t_end"] > t0 - pad and c["t_start"] < t1 + pad))


def main() -> int:
    w, m = load()
    pairs = []
    print(f"{'術語':16s} {'人工出現':>4s}  Whisper 在同一時段寫成什麼")
    print("─" * 78)
    for term in TERMS:
        hits = [c for c in m if term in c["text_raw"]]
        if not hits:
            continue
        cue = hits[0]
        span = whisper_span(w, cue["t_start"], cue["t_end"])
        ok = term in span
        # 從 span 中框出對應位置：用術語前後各 4 字當錨點做粗定位
        print(f"{term:16s} {len(hits):4d}  {'✓ 正確' if ok else span[:52]}")
        if not ok:
            pairs.append({"term": term, "whisper_span": span,
                          "t_start": cue["t_start"], "t_end": cue["t_end"],
                          "manual": cue["text_raw"], "count": len(hits)})
    (HERE / "term_pairs.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n人工有、Whisper 對不上的術語 {len(pairs)} 個 → term_pairs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
