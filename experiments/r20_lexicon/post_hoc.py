"""量詞庫用在**事後校正**的效果（而非 ASR 解碼）。

    conda run -n pipe-cpu python -m experiments.r20_lexicon.post_hoc

initial_prompt 那條路實測沒用（術語 4/19→5/19）。這裡測另一個插入點：
拿詞庫去**掃描 ASR 輸出**，找出「拼音很像某個術語、但寫得不一樣」的片段，
產生校正候選，再交給 R13 既有的閘驗證。

詞庫來源：`slide_text`（無字幕影片唯一拿得到的東西）。
標準答案：人工字幕。
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from weft.validation.corrections import pinyin_similarity, unauthorized_reason

HERE = Path(__file__).parent
WORK = Path("work/zIglvjoU9vo")
MIN_SIM = 0.70   # 候選門檻：拼音相似但不相同


def lexicon_from_slides() -> list[str]:
    """從 slide_text 抽出候選術語：純中文、2–6 字的連續片段。"""
    ir = json.loads((WORK / "08_video.json").read_text())
    terms: set[str] = set()
    for s in ir["slides"]:
        for line in (s.get("slide_text") or "").splitlines():
            for chunk in re.split(r"[^一-鿿]+", line):
                for n in (2, 3, 4, 5):
                    for i in range(len(chunk) - n + 1):
                        terms.add(chunk[i:i + n])
    return sorted(terms)


def main() -> int:
    manual = json.loads((WORK / "05_transcript.json").read_text())["cues"]
    asr = json.loads((HERE.parent / "r17_whisper" / "whisper_cues.json").read_text())
    from opencc import OpenCC
    cc = OpenCC("s2tw")
    asr_text = re.sub(r"\s", "", cc.convert("".join(c["text"] for c in asr)))
    manual_text = re.sub(r"\s", "", "".join(c["text_raw"] for c in manual))

    lex = lexicon_from_slides()
    print(f"從 slide_text 抽出候選詞 {len(lex)} 個")

    # 只看「詞庫裡有、但 ASR 沒寫出來」的術語——那才需要校正
    missing = [t for t in lex if t not in asr_text and len(t) >= 3]
    print(f"其中 ASR 沒寫出來的（≥3 字）：{len(missing)}")

    proposals = []
    for term in missing:
        n = len(term)
        best, best_sim = None, 0.0
        for i in range(len(asr_text) - n + 1):
            cand = asr_text[i:i + n]
            if cand == term:
                continue
            sim = pinyin_similarity(cand, term)
            if sim > best_sim:
                best, best_sim = cand, sim
        if best and best_sim >= MIN_SIM:
            proposals.append((best, term, best_sim))

    # 去重：同一個 ASR 片段只留最像的那個提案
    by_src: dict[str, tuple[str, str, float]] = {}
    for src, tgt, sim in proposals:
        if src not in by_src or sim > by_src[src][2]:
            by_src[src] = (src, tgt, sim)
    proposals = sorted(by_src.values(), key=lambda p: -p[2])

    passed = [p for p in proposals if unauthorized_reason(p[0], p[1]) is None]
    correct = [p for p in passed if p[1] in manual_text and p[0] not in manual_text]

    print(f"\n產生校正候選 {len(proposals)} 筆")
    print(f"  通過 R13 三道閘：{len(passed)}")
    print(f"  其中確實是對的（正確寫法在人工字幕中、錯的不在）：{len(correct)}")
    if passed:
        print(f"  precision = {len(correct)}/{len(passed)} = {len(correct)/len(passed):.0%}")

    print("\n逐門檻的精確度：")
    print(f"{'拼音門檻':>8s} {'候選':>5s} {'判對':>5s} {'precision':>10s}")
    for thr in (0.70, 0.80, 0.90, 0.95, 1.00):
        sub = [p for p in passed if p[2] >= thr]
        good = [p for p in sub if p in correct]
        pr = f"{len(good)/len(sub):.0%}" if sub else "—"
        print(f"{thr:8.2f} {len(sub):5d} {len(good):5d} {pr:>10s}")

    print("\n判定正確的前 12 筆：")
    for src, tgt, sim in correct[:12]:
        print(f"  {src} → {tgt}   拼音 {sim:.2f}")
    print("\n通過閘但判定不正確的前 8 筆（誤報）：")
    for p in passed:
        if p not in correct:
            print(f"  {p[0]} → {p[1]}   拼音 {p[2]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
