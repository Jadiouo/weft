"""把逐字稿來源換成 Whisper，量 §5.4 溯源通過率的實際變化。

    conda run -n pipe-cpu python -m experiments.r18_whisper_impact.provenance_swap

**隔離的問題**：如果來源文字換成 Whisper 的（而不是人工字幕的），
既有的 content_block 還找得到依據嗎？

三組對照：
  A. 人工字幕（現況基準，98.6%）
  B. Whisper 原樣（**簡體**）
  C. Whisper 轉繁（s2tw）

B 與 C 的差距就是**繁簡造成的**；A 與 C 的差距是**辨識錯誤造成的**。
兩者分開才知道該修哪個。

**這不是完整模擬**——真實情況下 VLM 看到 Whisper 逐字稿會產生**不同的**
block。這裡固定 block、只換來源，隔離「來源品質」單一變因。
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from opencc import OpenCC

from weft.config import Config
from weft.ir import VerificationStatus, VideoIR
from weft.validation.provenance import check_block, resolve_source

HERE = Path(__file__).parent
WORK = Path("work/zIglvjoU9vo")
CC = OpenCC("s2tw")


def whisper_text_for(cues, t0: float, t1: float, convert: bool) -> str:
    s = "".join(c["text"] for c in cues if c["t_end"] > t0 and c["t_start"] < t1)
    return CC.convert(s) if convert else s


def run(ir: VideoIR, cfg, source_for) -> tuple[float, Counter]:
    """與 `check_video` 同一套判定，含 degenerate_copy 的型別級後處理——
    少了那一步數字會偏嚴，與管線報的通過率對不上。"""
    verdicts = []
    copy_counts: dict[str, list[int]] = {}
    for seg in ir.segments:
        if seg.understanding is None:
            continue
        transcript = source_for(seg)
        for i, block in enumerate(seg.understanding.content_blocks):
            source = resolve_source(ir, block, transcript)
            v = check_block(block, source, cfg, seg.segment_id, i)
            verdicts.append((str(block.type), v))
            bucket = copy_counts.setdefault(str(block.type), [0, 0])
            bucket[1] += 1
            if v.copy_ratio >= cfg.copy_similarity:
                bucket[0] += 1
    for _t, v in verdicts:
        if v.status is not VerificationStatus.DEGENERATE_COPY:
            continue
        copied, total_t = copy_counts[v.content_type]
        if total_t and copied / total_t <= cfg.max_copy_ratio.get(v.content_type, 0.8):
            v.status = VerificationStatus.VERIFIED
    stats = Counter(f"{t}/{v.status.value}" for t, v in verdicts)
    ok = sum(1 for _t, v in verdicts if v.status is VerificationStatus.VERIFIED)
    return (ok / len(verdicts) if verdicts else 0.0), stats


def main() -> int:
    cfg = Config().provenance
    ir = VideoIR.model_validate_json((WORK / "08_video.json").read_text())
    cues = json.loads((HERE.parent / "r17_whisper" / "whisper_cues.json").read_text())

    arms = {
        "A 人工字幕（基準）": lambda s: s.transcript_corrected or s.transcript_raw,
        "B Whisper 原樣（簡體）": lambda s: whisper_text_for(cues, s.t_start, s.t_end, False),
        "C Whisper 轉繁": lambda s: whisper_text_for(cues, s.t_start, s.t_end, True),
    }

    print(f"{'組別':24s} {'溯源通過率':>9s}")
    print("─" * 40)
    results = {}
    for name, fn in arms.items():
        rate, stats = run(ir, cfg, fn)
        results[name] = (rate, stats)
        print(f"{name:24s} {rate:9.1%}")

    print("\n逐型別通過率：")
    types = ["經文原文", "白話解說", "口頭延伸", "圖表描述"]
    print(f"{'型別':10s} " + " ".join(f"{k[:1]}組".rjust(7) for k in arms))
    for t in types:
        cells = []
        for name in arms:
            _, st = results[name]
            v = st.get(f"{t}/verified", 0)
            n = sum(c for k, c in st.items() if k.startswith(f"{t}/"))
            cells.append(f"{v}/{n}".rjust(7))
        print(f"{t:10s} " + " ".join(cells))

    a = results["A 人工字幕（基準）"][0]
    b = results["B Whisper 原樣（簡體）"][0]
    c = results["C Whisper 轉繁"][0]
    print(f"\n繁簡造成的落差（C−B）：{c - b:+.1%}")
    print(f"辨識錯誤造成的落差（C−A）：{c - a:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
