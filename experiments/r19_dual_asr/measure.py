"""量「兩套 ASR 不一致 → 定位術語錯誤」這個假設。

    conda run -n pipe-cpu python -m experiments.r19_dual_asr.measure

R17 §策略 3 列的第 2 個未解問題：**不一致 ≠ 錯在哪裡。**
兩套 ASR 可能在同一個罕見術語上**一致地錯**，也可能在贅詞上大量無意義地
不一致。這裡用人工字幕當標準答案，把那個假設量出來。

判準：對每個在人工字幕中出現的術語，取其時間窗，看兩套 ASR 在同一窗內
有沒有寫出該術語。四種情形：

  兩者都對         → 沒事
  只有一邊錯       → **交叉檢查抓得到**
  兩者都錯但寫得不同 → **交叉檢查抓得到**
  兩者都錯且寫得一樣 → **交叉檢查抓不到**（最危險：看起來像互相印證）
"""
from __future__ import annotations
import json, re
from pathlib import Path

HERE = Path(__file__).parent
WORK = Path("work/zIglvjoU9vo")
PAD = 2.5  # 時間窗左右放寬，吸收兩套 ASR 的分段差異


def spans(cues, t0, t1, key="text"):
    return re.sub(r"\s+", "", "".join(
        c[key] for c in cues if c["t_end"] > t0 - PAD and c["t_start"] < t1 + PAD))


def main() -> int:
    large = json.loads((HERE.parent / "r17_whisper" / "whisper_cues.json").read_text())
    # r17 的輸出未轉繁，這裡補上，才與 medium 同基準
    from opencc import OpenCC
    cc = OpenCC("s2tw")
    large = [{**c, "text": cc.convert(c["text"])} for c in large]
    medium = json.loads((HERE / "cues_medium.json").read_text())
    manual = json.loads((WORK / "05_transcript.json").read_text())["cues"]

    ir = json.loads((WORK / "08_video.json").read_text())
    terms = sorted({t for t in ir.get("term_index", []) if isinstance(t, str) and len(t) >= 2},
                   key=len, reverse=True)

    both_ok = one_wrong = both_diff = both_same = 0
    invisible = []
    for term in terms:
        hits = [c for c in manual if term in c["text_raw"]]
        if not hits:
            continue
        cue = hits[0]
        sl = spans(large, cue["t_start"], cue["t_end"])
        sm = spans(medium, cue["t_start"], cue["t_end"])
        hl, hm = term in sl, term in sm
        if hl and hm:
            both_ok += 1
        elif hl != hm:
            one_wrong += 1
        else:
            # 兩者都沒寫出這個術語——它們寫的是同一個錯的東西嗎？
            if sl == sm:
                both_same += 1
                invisible.append((term, sl[:40]))
            else:
                both_diff += 1

    checked = both_ok + one_wrong + both_diff + both_same
    errors = one_wrong + both_diff + both_same
    catchable = one_wrong + both_diff
    print(f"受檢術語（在人工字幕中出現過）：{checked}")
    print(f"  兩者都對            {both_ok:4d}")
    print(f"  只有一邊錯          {one_wrong:4d}   ← 交叉檢查抓得到")
    print(f"  兩者都錯、寫得不同  {both_diff:4d}   ← 交叉檢查抓得到")
    print(f"  兩者都錯、寫得一樣  {both_same:4d}   ← **抓不到**")
    if errors:
        print(f"\n錯誤總數 {errors}，交叉檢查的 recall = {catchable}/{errors} = {catchable/errors:.0%}")
    if checked:
        print(f"不一致率（會被送去人工複核的比例）= "
              f"{(one_wrong+both_diff)/checked:.0%}")
    print("\n兩者一致地錯（最危險，看起來像互相印證）：")
    for t, s in invisible[:8]:
        print(f"  {t}  → 兩套都寫成：…{s}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
