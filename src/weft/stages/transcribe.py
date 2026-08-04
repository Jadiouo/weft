"""字幕解析與 ASR。SDD §4.2。

字幕優先序：手動 > 自動 > 無。手動字幕是人打的，品質遠高於任何 ASR，
而且免費、瞬間完成——SDD §4.2 把它排在 Whisper 之前是對的。
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_TIMING = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
_TAG = re.compile(r"<[^>]+>")


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path: Path) -> list[tuple[float, float, str]]:
    """解析 WebVTT，回傳 `(t_start, t_end, text)`。

    YouTube 的**自動**字幕會逐字滾動，同一句話在連續的 cue 中反覆出現、
    每次多幾個字。直接採用會讓逐字稿嚴重重複，所以此處合併連續且內容
    互為前綴的 cue，只保留最完整的版本。
    """
    raw: list[tuple[float, float, str]] = []
    start = end = 0.0
    buffer: list[str] = []

    def flush() -> None:
        text = _TAG.sub("", " ".join(buffer)).strip()
        text = html.unescape(re.sub(r"\s+", " ", text))
        if text:
            raw.append((start, end, text))
        buffer.clear()

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        timing = _TIMING.search(stripped)
        if timing:
            flush()
            g = timing.groups()
            start, end = _seconds(*g[:4]), _seconds(*g[4:])
        elif stripped and not stripped.startswith(("WEBVTT", "NOTE", "STYLE", "Kind:", "Language:")):
            if stripped.isdigit() and not buffer:
                continue  # SRT 風格的序號
            buffer.append(stripped)
    flush()

    return _dedupe_rolling(raw)


def _dedupe_rolling(cues: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """合併「逐字滾動」的重複 cue。

    判準是前綴包含：若後一句以前一句開頭（去空白後），代表它是同一句的
    更完整版本，取代之並延長時間區間。
    """
    out: list[tuple[float, float, str]] = []
    for start, end, text in cues:
        if out:
            prev_start, _prev_end, prev_text = out[-1]
            a, b = prev_text.replace(" ", ""), text.replace(" ", "")
            if b.startswith(a) and len(b) > len(a):
                out[-1] = (prev_start, end, text)
                continue
            if a == b:
                out[-1] = (prev_start, end, prev_text)
                continue
        out.append((start, end, text))
    return out


def whisper_transcribe(video: Path, cfg, initial_prompt: str | None) -> list[tuple[float, float, str]]:
    """faster-whisper。SDD §4.2 策略 2。

    失敗行為（§4.2）：OOM → 降 batch size 重試一次 → 仍失敗則往上拋，
    由呼叫端標記該影片 failed 並繼續下一支。
    """
    from faster_whisper import WhisperModel

    p = cfg.s1a
    attempts = [
        {"device": p.device, "compute_type": p.compute_type},
        # 降級重試：int8 量化大幅降低 VRAM 需求
        {"device": p.device, "compute_type": "int8_float16"},
    ]
    last: Exception | None = None
    for i, kwargs in enumerate(attempts):
        try:
            model = WhisperModel(p.whisper_model, **kwargs)
            segments, _info = model.transcribe(
                str(video),
                language=p.language,
                beam_size=p.beam_size,
                vad_filter=p.vad_filter,
                initial_prompt=initial_prompt,
            )
            return [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
        except RuntimeError as exc:
            last = exc
            if "out of memory" not in str(exc).lower() or i == len(attempts) - 1:
                raise
            log.warning("Whisper OOM，改用 %s 重試一次（§4.2）", attempts[i + 1]["compute_type"])
        finally:
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()  # §8：large-v3 與 Qwen3-VL 不可同時常駐
            except ImportError:
                pass
    raise RuntimeError(f"Whisper 轉錄失敗：{last}")
