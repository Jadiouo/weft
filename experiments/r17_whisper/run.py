"""R17：實跑 S1a 策略 2（Whisper），對人工字幕量退化。

    conda run -n pipe-gpu python -m experiments.r17_whisper.run

`zIglvjoU9vo` 有**人工**字幕（zh-Hant，925 cues），是天然的標準答案。
在同一支影片上跑 Whisper，就能量出「沒有逐字稿時我們會損失什麼」。

呼叫的是管線裡真正的 `whisper_transcribe()`——那條路從未執行過，
跑它本身就是第一次測試。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from weft.config import Config
from weft.stages.transcribe import whisper_transcribe

def main() -> int:
    cfg = Config()
    video = Path("work/zIglvjoU9vo/01_video.mp4")
    t0 = time.time()
    cues = whisper_transcribe(video, cfg, initial_prompt=None)
    dt = time.time() - t0
    out = Path(__file__).parent / "whisper_cues.json"
    out.write_text(json.dumps(
        [{"t_start": a, "t_end": b, "text": t} for a, b, t in cues],
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成：{len(cues)} 個 cue，耗時 {dt/60:.1f} 分 → {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
