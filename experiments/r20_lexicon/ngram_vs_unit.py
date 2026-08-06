"""詞庫該用「任意 n-gram」還是「標點分隔的完整單位」？

    conda run -n pipe-cpu python -m experiments.r20_lexicon.ngram_vs_unit

**起因**：n-gram 版在真實資料上把正確文字改壞——`一開始`→`以開始`、
`任何一`→`任何意`、`生命之`→`生命只`。那些「術語」只是投影片某行的碎片。

兩組詞庫、兩份逐字稿（Whisper 與人工），四格都要看：
人工字幕幾乎沒有術語錯誤，**任何提案都是誤報**，是最嚴的 precision 測試。
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from opencc import OpenCC
from weft.validation.corrections import pinyin_similarity, unauthorized_reason

WORK = Path("work/zIglvjoU9vo")
HERE = Path(__file__).parent
HAN = re.compile(r"[一-鿿]+")


def slide_lines() -> list[str]:
    ir = json.loads((WORK / "08_video.json").read_text())
    out = []
    for s in ir["slides"]:
        out += (s.get("slide_text") or "").splitlines()
    return out


def lex_ngram(lo=3, hi=5) -> set[str]:
    terms = set()
    for line in slide_lines():
        for run in HAN.findall(line):
            for n in range(lo, hi + 1):
                for i in range(len(run) - n + 1):
                    terms.add(run[i:i + n])
    return terms


def lex_unit(lo=3, hi=12) -> set[str]:
    """標點／空白分隔的完整片段——不是碎片。"""
    return {run for line in slide_lines() for run in HAN.findall(line)
            if lo <= len(run) <= hi}


def scan(text: str, lexicon: set[str], thr=0.90) -> list[tuple[str, str, float]]:
    flat = re.sub(r"\s+", "", text)
    best: dict[str, tuple[str, str, float]] = {}
    for term in lexicon:
        if term in flat:
            continue
        n = len(term)
        for i in range(len(flat) - n + 1):
            c = flat[i:i + n]
            if c == term:
                continue
            s = pinyin_similarity(c, term)
            if s >= thr and (c not in best or s > best[c][2]):
                best[c] = (c, term, s)
    return [p for p in best.values() if unauthorized_reason(p[0], p[1]) is None]


def main() -> int:
    manual_cues = json.loads((WORK / "05_transcript.json").read_text())["cues"]
    manual = re.sub(r"\s", "", "".join(c["text_raw"] for c in manual_cues))
    cc = OpenCC("s2tw")
    whisper = re.sub(r"\s", "", cc.convert("".join(
        c["text"] for c in json.loads(
            (HERE.parent / "r17_whisper" / "whisper_cues.json").read_text()))))

    for name, lex in (("任意 n-gram (3–5)", lex_ngram()), ("標點分隔的完整單位", lex_unit())):
        print(f"\n{'='*66}\n{name}：詞庫 {len(lex)} 詞\n{'='*66}")

        # 人工字幕：幾乎沒有術語錯誤，任何提案都是誤報
        fp = scan(manual, lex)
        print(f"人工字幕上的提案（≈全為誤報）：{len(fp)}")
        for p in fp[:6]:
            print(f"    {p[0]} → {p[1]}")

        # Whisper：提案中「正確寫法在人工字幕、錯的不在」才算對
        props = scan(whisper, lex)
        good = [p for p in props if p[1] in manual and p[0] not in manual]
        pr = f"{len(good)/len(props):.0%}" if props else "—"
        print(f"Whisper 上的提案：{len(props)}，判對 {len(good)}，precision {pr}")
        keep = []
        for s, t, _v in sorted(good, key=lambda x: -len(x[0])):
            if not any(s in k and t in v for k, v in keep):
                keep.append((s, t))
        print(f"  去重後的相異錯誤：{len(keep)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
