"""A/B：Whisper 加不加 slide_text 建的 initial_prompt。

    conda run -n pipe-gpu python -m experiments.r20_lexicon.run

對照組是 R17 的無 prompt 輸出（同模型、同參數）。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from weft.config import Config
from weft.stages.transcribe import to_traditional, whisper_transcribe

HERE = Path(__file__).parent


def main() -> int:
    cfg = Config()
    prompt = (HERE / "initial_prompt.txt").read_text(encoding="utf-8")
    t0 = time.time()
    rows = whisper_transcribe(Path("work/zIglvjoU9vo/01_video.mp4"), cfg, prompt)
    rows = to_traditional(rows, cfg.s1a.asr_script_conversion)
    out = HERE / "cues_with_prompt.json"
    out.write_text(json.dumps(
        [{"t_start": a, "t_end": b, "text": t} for a, b, t in rows],
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} cue，{(time.time()-t0)/60:.1f} 分 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
