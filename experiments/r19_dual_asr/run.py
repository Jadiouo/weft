"""跑第二個 ASR，供雙重驗證的可行性量測。

    conda run -n pipe-gpu python -m experiments.r19_dual_asr.run [模型]

SDD 策略 3 寫的是「YouTube 自動字幕 + Whisper 交叉檢查」，但實測這個語料
**根本沒有獨立的 ASR 軌**（見 REPORT §1）。所以改用第二個本地 ASR。

**medium 與 large-v3 同家族，不是真正獨立。** 這是刻意的保守選擇：
同家族會**共享錯誤模式**，所以「兩者都錯」的比例是真正獨立系統的**上界**，
而「不一致能抓到的錯誤」是**下界**。若連這樣都顯示訊號無效，
換成獨立系統也未必救得回來——反之則不能直接推論。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from weft.config import Config
from weft.stages.transcribe import to_traditional, whisper_transcribe


def main(argv: list[str]) -> int:
    model = argv[0] if argv else "medium"
    cfg = Config()
    cfg.s1a.whisper_model = model
    t0 = time.time()
    rows = whisper_transcribe(Path("work/zIglvjoU9vo/01_video.mp4"), cfg, None)
    rows = to_traditional(rows, cfg.s1a.asr_script_conversion)
    out = Path(__file__).parent / f"cues_{model.replace('/', '_')}.json"
    out.write_text(json.dumps(
        [{"t_start": a, "t_end": b, "text": t} for a, b, t in rows],
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{model}：{len(rows)} cue，{(time.time()-t0)/60:.1f} 分 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
