"""R32（票 10）：逐段局部 prompt vs R20 的全域 prompt。

R20 的結論是「解碼層沒用」，但 DocWhisper 用**同一個插入點**拿到
WER 相對改善 14.3%。差別在實作：

| | R20 | DocWhisper |
|---|---|---|
| 粒度 | 整支影片一個**全域** prompt | **逐 utterance** 只餵當下那張投影片 |
| 格式 | 經文原文段落 | 詞序列 `word 1, word 2, ...` |

**這張票必須排在票 08 之後**：文獻指出 misaligned slides 會讓結果**變差**，
分段沒修好之前「逐段」本身就是錯的。v0.5 的分段召回是 0.93（v0.4 是 0.60）。

四組對照，前兩組已經跑過：

| 組 | prompt | 來源 |
|---|---|---|
| A | 無 | `experiments/r17_whisper/whisper_cues.json` |
| B | 全域 200 字經文段落 | `experiments/r20_lexicon/cues_with_prompt.json` |
| C | **逐段詞序列**（DocWhisper 式） | 這支腳本 |
| D | **逐段經文段落** | 這支腳本（隔離「格式」與「粒度」）|

C 與 D 的差別只有格式，C 與 B 的差別只有粒度——三組兩兩比才分得出
是粒度有用還是格式有用。

    conda run -n pipe-gpu python -m experiments.r32_local_prompt.run
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

HERE = Path(__file__).parent
VIDEO_ID = "zIglvjoU9vo"
WORK = Path("work") / VIDEO_ID

#: 詞序列的分隔符。DocWhisper 用 `word 1, word 2, ...`。
SEP = "、"


def slide_terms(slide_text: str, max_terms: int = 60) -> list[str]:
    """把投影片文字切成詞序列。

    **用標點切，不用 n-gram**——D25 已經量過：n-gram 在錯誤基率高的資料上
    precision 看似 91%，接到真實管線卻把正確文字改壞。這裡雖然只是餵給
    解碼器不是直接改字，但同一個道理：切出來的東西要是真的詞。
    """
    parts = re.split(r"[\s，。、；：！？,.!?;:\n]+", slide_text or "")
    seen: list[str] = []
    for p in parts:
        p = p.strip()
        if 2 <= len(p) <= 12 and p not in seen:
            seen.append(p)
    return seen[:max_terms]


def build_prompts(mode: str) -> list[tuple[float, float, str]]:
    """逐段的 `(起, 迄, prompt)`。`mode` 是 `words` 或 `prose`。"""
    segs = json.loads((WORK / "06_segments.json").read_text(encoding="utf-8"))
    ir = json.loads((WORK / "08_video.json").read_text(encoding="utf-8"))
    slides = {s["slide_id"]: s for s in ir["slides"]}

    out = []
    for s in segs:
        ref = s.get("slide_ref") or s.get("candidate_ref")
        text = (slides.get(ref, {}) or {}).get("slide_text") or ""
        if mode == "words":
            terms = slide_terms(text)
            prompt = SEP.join(terms) if terms else ""
        else:
            prompt = " ".join(text.split())
        out.append((s["t_start"], s["t_end"], prompt))
    return out


def _slice_audio(video: Path, t0: float, t1: float, dest: Path) -> bool:
    """抽出一段音訊。ffmpeg 失敗時回傳 False 而不是丟例外——
    一段抽不出來不該讓整輪實驗停下。"""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(dest)],
        capture_output=True, check=False)
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 1000


def main() -> int:
    from faster_whisper import WhisperModel

    from weft.config import Config
    from weft.stages.transcribe import to_traditional

    cfg = Config()
    video = WORK / "01_video.mp4"
    model = WhisperModel(cfg.s1a.whisper_model, device=cfg.s1a.device,
                         compute_type=cfg.s1a.compute_type)
    tmp = HERE / "_audio"
    tmp.mkdir(exist_ok=True)

    for mode in ("words", "prose"):
        dest = HERE / f"cues_local_{mode}.json"
        if dest.exists():
            print(f"{dest.name} 已存在，跳過")
            continue
        rows: list[tuple[float, float, str]] = []
        t0 = time.time()
        prompts = build_prompts(mode)
        for i, (a, b, prompt) in enumerate(prompts):
            chunk = tmp / f"{i:03d}.wav"
            if not _slice_audio(video, a, b, chunk):
                print(f"  段 {i} 抽音訊失敗，跳過")
                continue
            segments, _ = model.transcribe(
                str(chunk), language=cfg.s1a.language, beam_size=cfg.s1a.beam_size,
                vad_filter=cfg.s1a.vad_filter,
                initial_prompt=prompt or None,
            )
            for s in segments:
                rows.append((a + s.start, a + s.end, s.text))
            chunk.unlink(missing_ok=True)
        rows = to_traditional(rows, cfg.s1a.asr_script_conversion)
        dest.write_text(json.dumps(
            [{"t_start": x, "t_end": y, "text": t} for x, y, t in rows],
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{mode}: {len(rows)} cue，{(time.time() - t0) / 60:.1f} 分 → {dest.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
