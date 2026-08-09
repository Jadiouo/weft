"""R39 §4 的後續：`provenance_kind` 判成 transcript，但內容其實來自投影片？

手讀時三段裡出現一次（`xBfyWwYylSA#024#b01`）。這裡量基率。

**對照組是通過組**：如果通過的 block 也普遍「對投影片的覆蓋高於逐字稿」，
那這把尺只是在說「投影片文字比較短所以容易命中」，不是在說歸屬錯。
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from measure import VIDEOS, coverage  # noqa: E402

from weft.config import Config  # noqa: E402
from weft.ir import VideoIR  # noqa: E402
from weft.paths import WorkPaths  # noqa: E402


def main() -> None:
    cfg = Config.load("configs/local.yaml")
    rows = []
    for vid in VIDEOS:
        work = WorkPaths(cfg.work_dir, vid)
        if not work.video_ir.exists():
            continue
        ir = VideoIR.model_validate_json(work.video_ir.read_text(encoding="utf-8"))
        cues = json.loads(work.transcript.read_text(encoding="utf-8"))["cues"]
        slides = {s.slide_id: (s.slide_text or "") for s in ir.slides}

        for seg in ir.segments:
            if seg.understanding is None:
                continue
            raw = "".join(cues[i]["text_raw"] for i in seg.cue_indices if i < len(cues))
            slide_text = slides.get(seg.slide_ref or "", "")
            for bi, block in enumerate(seg.understanding.content_blocks):
                if block.verification is None:
                    continue
                lit_tr, _ = coverage(block.text, raw)
                lit_sl, _ = coverage(block.text, slide_text) if slide_text else (0.0, 0.0)
                rows.append({
                    "video_id": vid,
                    "block": f"{seg.segment_id}#b{bi:02d}",
                    "status": block.verification.value,
                    "kind": block.provenance.kind.value,
                    "has_slide": bool(slide_text),
                    "cov_transcript": round(lit_tr, 3),
                    "cov_slide": round(lit_sl, 3),
                })

    out = pathlib.Path(__file__).with_name("source_kind.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    withslide = [r for r in rows if r["has_slide"]]
    print(f"{len(rows)} 個 block，其中 {len(withslide)} 個所屬段落有投影片文字\n")
    print(f"{'組別':<10}{'型別':<12}{'n':>5}{'逐字稿覆蓋':>11}{'投影片覆蓋':>11}"
          f"{'投影片較高':>11}")
    for st in ("unverified", "verified"):
        for kind in ("transcript", "slide_ocr"):
            g = [r for r in withslide if r["status"] == st and r["kind"] == kind]
            if not g:
                continue
            m = lambda k: sum(r[k] for r in g) / len(g)
            higher = sum(1 for r in g if r["cov_slide"] > r["cov_transcript"]) / len(g)
            label = "未通過" if st == "unverified" else "通過"
            print(f"{label:<10}{kind:<12}{len(g):>5}{m('cov_transcript'):>11.3f}"
                  f"{m('cov_slide'):>11.3f}{higher:>11.1%}")


if __name__ == "__main__":
    main()
