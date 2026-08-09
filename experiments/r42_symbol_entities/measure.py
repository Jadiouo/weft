"""R42：具名實體檢查在 STEM 素材上是空轉的——量一個補上符號的版本。

實測：`extract_named_entities("四乘四矩陣用於表示坐標系B到坐標系A的完整轉換")`
回傳**空集合**。現行的抽取器只認書名、年代、中文數字（為中醫講經設計），
所以在機器人學素材上它一個實體都抽不到，等於沒有在檢查。

而 R39 §3.1 手讀時抓到的正是這一類：`UiKi5#023#b01` 寫出了
`R₁₁(a)`、`R₁₂(β)`，**講者從頭到尾沒說過**（他說「對哪個軸轉幾度」）。
符號感知的檢查本來就該抓到它。

**必須量誤報**：模型會把口說的「四乘四」寫成 `4x4`、「R 一一」寫成 `R₁₁`，
那是正規化不是編造（R36）。所以要有變體正規化，而且要拿
**通過組當對照**——新檢查若在通過組上也大量告警，它就是誤報機器。

D25 的教訓：在錯誤基率高的資料上量出的 precision 會虛高。
所以這裡報的是**兩組的告警率差**，不是單一組的準確率。
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from weft.config import Config  # noqa: E402
from weft.ir import Transcript, VerificationStatus, VideoIR  # noqa: E402
from weft.paths import WorkPaths  # noqa: E402
from weft.validation.provenance import resolve_source  # noqa: E402

VIDEOS = ("UiKi5-Arce4", "xBfyWwYylSA", "jgVD7IPNTs8", "cxrqHABhWOU",
          "2FjApOVIbUs", "cvb4Bl93lzQ", "zIglvjoU9vo", "C_CFyilE-ks")

#: 下標數字 → 一般數字，方便比對
_SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_CJK_DIGIT = {"零": "0", "一": "1", "二": "2", "兩": "2", "三": "3", "四": "4",
              "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}

#: 符號型實體。刻意**不抓單獨的大寫字母**——「A 乘 B」裡的 A、B 太常見，
#: 抓了會把每個 block 都告警。只抓帶結構的。
#:
#: **順序有意義**：長的樣式先吃掉字元，否則 `4x4` 會被切出一個假的 `x4`
#: （第一版就是這樣，通過組的誤報有一半來自它）。
_PATTERNS = (
    re.compile(r"[A-Za-z]{2,}\([^)]{1,20}\)"),  # Rz(φ)、Euler(φ,θ,ψ)
    re.compile(r"\d{1,3}\s*[x×]\s*\d{1,3}"),    # 4x4、3×3
    re.compile(r"[A-Za-z][₀-₉ₓ]{1,3}"),          # R₁₁、n₃、nₓ
    re.compile(r"(?<![A-Za-z\d])[A-Za-z]_?\d{1,3}(?![\d.])"),  # R11、T_2
)


def symbol_entities(text: str) -> set[str]:
    """不重疊抽取：長樣式優先，吃掉的字元不再參與後面的樣式。"""
    taken = [False] * len(text)
    out: set[str] = set()
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            a, b = m.span()
            if any(taken[a:b]):
                continue
            for i in range(a, b):
                taken[i] = True
            e = m.group(0).strip()
            if e:
                out.add(e)
    return out


def variants(entity: str) -> set[str]:
    """同一個符號的其他寫法。**沒有這個，正規化會被誤判成編造。**"""
    e = entity.translate(_SUBSCRIPT)
    forms = {entity, e, e.replace("_", ""), e.replace(" ", "")}

    m = re.fullmatch(r"(\d{1,3})\s*[x×]\s*(\d{1,3})", e.replace(" ", ""))
    if m:
        a, b = m.group(1), m.group(2)
        cjk = {v: k for k, v in _CJK_DIGIT.items()}
        ca, cb = cjk.get(a, a), cjk.get(b, b)
        forms |= {f"{a}x{b}", f"{a}×{b}", f"{a}乘{b}", f"{ca}乘{cb}",
                  f"{ca}x{cb}", f"{ca}乘以{cb}", f"{a}乘以{b}"}

    m = re.fullmatch(r"([A-Za-z])(\d{1,3})", e)
    if m:
        head, digits = m.group(1), m.group(2)
        spoken = "".join({v: k for k, v in _CJK_DIGIT.items()}.get(d, d) for d in digits)
        forms |= {f"{head}{digits}", f"{head} {digits}", f"{head}{spoken}",
                  f"{head} {spoken}"}
    return {f for f in forms if f}


def unsupported(text: str, source: str) -> list[str]:
    src = re.sub(r"\s+", "", source)
    src_norm = src.translate(_SUBSCRIPT)
    bad = []
    for e in symbol_entities(text):
        if not any(v.replace(" ", "") in src or v.replace(" ", "") in src_norm
                   for v in variants(e)):
            bad.append(e)
    return sorted(bad)


def main() -> None:
    cfg = Config.load("configs/local.yaml")
    rows = []
    for vid in VIDEOS:
        w = WorkPaths(cfg.work_dir, vid)
        if not w.video_ir.exists():
            continue
        ir = VideoIR.model_validate_json(w.video_ir.read_text(encoding="utf-8"))
        tr = Transcript.model_validate_json(w.transcript.read_text(encoding="utf-8"))
        cues = tr.cues
        for seg in ir.segments:
            if seg.understanding is None:
                continue
            # `resolve_source` 的第三個參數是**該段的逐字稿字串**，不是
            # Transcript 物件。這裡自己組。
            seg_text = "".join(cues[i].text_raw for i in seg.cue_indices
                               if i < len(cues))
            for bi, b in enumerate(seg.understanding.content_blocks):
                if b.verification is None:
                    continue
                src = resolve_source(ir, b, seg_text)
                ents = symbol_entities(b.text)
                miss = unsupported(b.text, src)
                rows.append({
                    "video_id": vid, "block": f"{seg.segment_id}#b{bi:02d}",
                    "status": b.verification.value, "type": b.type.value,
                    "kind": b.provenance.kind.value,
                    "n_symbols": len(ents), "n_unsupported": len(miss),
                    "unsupported": miss[:6], "text": b.text[:100],
                })

    pathlib.Path(__file__).with_name("results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    stem = {"UiKi5-Arce4", "xBfyWwYylSA", "jgVD7IPNTs8"}
    print(f"{len(rows)} 個 block\n")
    print(f"{'素材':<8}{'組別':<8}{'n':>5}{'有符號':>8}{'告警':>7}{'告警率':>9}")
    for name, sel in (("STEM", lambda r: r["video_id"] in stem),
                      ("其他", lambda r: r["video_id"] not in stem)):
        for st, label in (("unverified", "未通過"), ("verified", "通過")):
            g = [r for r in rows if sel(r) and r["status"] == st]
            if not g:
                continue
            has = sum(1 for r in g if r["n_symbols"] > 0)
            flag = sum(1 for r in g if r["n_unsupported"] > 0)
            print(f"{name:<8}{label:<8}{len(g):>5}{has:>8}{flag:>7}"
                  f"{flag / len(g):>9.1%}")

    print("\n未通過且被新檢查告警的（前 8 筆）：")
    for r in [r for r in rows if r["status"] == "unverified" and r["n_unsupported"]][:8]:
        print(f"  {r['block']:<22}{str(r['unsupported']):<34}{r['text'][:44]}")
    print("\n**通過卻被告警的（誤報候選，前 8 筆）：**")
    for r in [r for r in rows if r["status"] == "verified" and r["n_unsupported"]][:8]:
        print(f"  {r['block']:<22}{str(r['unsupported']):<34}{r['text'][:44]}")


if __name__ == "__main__":
    main()
