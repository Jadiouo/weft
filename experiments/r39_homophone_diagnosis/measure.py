"""R39：溯源失敗裡有多少是「S4c 悄悄改對了 ASR 的同音錯字」（R27）。

R27 記過一個例子：Whisper 把「座標」聽成「做標」，S4c 寫 block 時
自己改對了，於是對 `transcript_raw` 的比對失敗——**判定是對的**，
但成因被歸進「真的溯不到」，而那一欄正是拿來判斷內容有沒有問題的。

這裡量它佔多少。**必須有對照組**：把同一套拼音比對套到
「block 對上**別的段落**的逐字稿」，看它會不會一樣說「找到了」。
對照組出錯或缺席時會偽裝成訊號（R23／R37 各栽過一次）。
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from weft.config import Config  # noqa: E402
from weft.ir import VerificationStatus, VideoIR  # noqa: E402
from weft.paths import WorkPaths  # noqa: E402
from weft.validation.corrections import pinyin_key  # noqa: E402

VIDEOS = ("UiKi5-Arce4", "xBfyWwYylSA", "zIglvjoU9vo", "C_CFyilE-ks",
          "cxrqHABhWOU", "2FjApOVIbUs", "cvb4Bl93lzQ", "jgVD7IPNTs8")
NGRAM = 3  # 以 3 字為單位比對


def han_ngrams(text: str, n: int = NGRAM) -> list[str]:
    """只取漢字，切成 n 字窗。標點與英數不參與——它們不是同音錯字的載體。"""
    han = re.sub(r"[^一-鿿]", "", text)
    return [han[i:i + n] for i in range(len(han) - n + 1)]


def pinyin_set(text: str, n: int = NGRAM) -> set[str]:
    han = re.sub(r"[^一-鿿]", "", text)
    key = pinyin_key(han)
    return {"".join(key[i:i + n]) for i in range(len(key) - n + 1)}


def coverage(block_text: str, source: str) -> tuple[float, float]:
    """回傳 (字面覆蓋率, 拼音覆蓋率)。

    拼音明顯高於字面 ⇒ 內容在逐字稿裡**聽得到但寫不同字** ⇒ 同音替換。
    """
    grams = han_ngrams(block_text)
    if not grams:
        return 1.0, 1.0
    src_chars = re.sub(r"[^一-鿿]", "", source)
    literal = sum(1 for g in grams if g in src_chars) / len(grams)

    src_py = pinyin_set(source)
    block_py = ["".join(pinyin_key(g)) for g in grams]
    phonetic = sum(1 for g in block_py if g in src_py) / len(block_py)
    return literal, phonetic


def main() -> None:
    cfg = Config.load("configs/local.yaml")
    rows = []
    for vid in VIDEOS:
        work = WorkPaths(cfg.work_dir, vid)
        if not work.video_ir.exists():
            continue
        ir = VideoIR.model_validate_json(work.video_ir.read_text(encoding="utf-8"))
        transcript = json.loads(work.transcript.read_text(encoding="utf-8"))
        cues = transcript["cues"]

        for si, seg in enumerate(ir.segments):
            if seg.understanding is None:
                continue
            own = "".join(cues[i]["text_raw"] for i in seg.cue_indices
                          if i < len(cues))
            # **對照組**：同一支影片、離得最遠的另一段。內容無關，
            # 但語者、主題、ASR 特性都一樣——比隨機字串公道得多。
            far = max(range(len(ir.segments)), key=lambda j: abs(j - si))
            other_seg = ir.segments[far]
            other = "".join(cues[i]["text_raw"] for i in other_seg.cue_indices
                            if i < len(cues))

            for bi, block in enumerate(seg.understanding.content_blocks):
                if block.verification is None:
                    continue
                lit, pho = coverage(block.text, own)
                _, pho_ctrl = coverage(block.text, other)
                rows.append({
                    "video_id": vid,
                    "block": f"{seg.segment_id}#b{bi:02d}",
                    "status": block.verification.value,
                    "literal": round(lit, 3),
                    "phonetic": round(pho, 3),
                    "phonetic_control": round(pho_ctrl, 3),
                    "gain": round(pho - lit, 3),
                    "gain_control": round(pho_ctrl - lit, 3),
                    "text": block.text[:120],
                })

    out = pathlib.Path(__file__).with_name("results.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"寫出 {out}（{len(rows)} 個 block）")

    unver = [r for r in rows if r["status"] == VerificationStatus.UNVERIFIED.value]
    ver = [r for r in rows if r["status"] == VerificationStatus.VERIFIED.value]
    mean = lambda xs, k: sum(x[k] for x in xs) / len(xs) if xs else float("nan")
    print(f"\n{'組別':<12}{'n':>5}{'字面':>8}{'拼音':>8}{'增益':>8}"
          f"{'對照增益':>10}")
    for name, group in (("未通過", unver), ("通過", ver)):
        print(f"{name:<12}{len(group):>5}{mean(group,'literal'):>8.3f}"
              f"{mean(group,'phonetic'):>8.3f}{mean(group,'gain'):>8.3f}"
              f"{mean(group,'gain_control'):>10.3f}")


if __name__ == "__main__":
    main()
