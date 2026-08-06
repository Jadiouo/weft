"""S4b：以投影片術語為詞庫的事後校正（D25）。

v0.3 移除了「詞庫 → 拼音 → 產生校正」那整條鏈（R11、D12/D13），
理由是詞庫來自本地 OCR，而本地 OCR 被移除了。**現在來源不同了**：
`slide_text` 由 S4 的 VLM 從投影片讀出，是正確的繁體。

R20 量了兩個插入點：

| 插入點 | 實測 | 判定 |
|---|---|---|
| Whisper `initial_prompt` | 術語 4/19 → 5/19，CER 12.3% → 12.0% | **沒用** |
| **事後校正**（本模組） | 拼音門檻 0.90 下 precision **100%**，抓到 14 個相異錯誤 | 可用 |

對照組：VLM 在人工字幕上只產生 7 筆校正。ASR 逐字稿的錯誤遠多於此，
而 VLM 只在「它剛好注意到」時才提——這個模組把它換成**程式逐一掃過詞庫**。

**這不是把 v0.3 的詞庫鏈請回來。** 差別在：

| | v0.2 的詞庫鏈 | 本模組 |
|---|---|---|
| 詞庫來源 | 本地 OCR（PaddleOCR，輸出簡體，D12） | `slide_text`（VLM 讀出的繁體） |
| 斷詞 | jieba（詞典是簡體，D12） | 不斷詞，取標點分隔的完整片段 |
| 產出 | 直接改逐字稿 | 只**提案**，仍走 R13 的三道閘 |

最後一列是關鍵：本模組不繞過任何既有的把關。
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

#: 詞庫只取這個長度範圍的片段。
#: 下限 3：2 字的同音碎片太多，誤報爆量。
#: 上限 12：更長的整段不會被完整聽錯，而且拼音比對成本是長度的平方。
MIN_TERM_LEN = 3
MAX_TERM_LEN = 12

#: 只保留連續中文。標點與英數在拼音比對上沒有意義。
_HAN_RUN = re.compile(r"[一-鿿]+")


def build_lexicon(slides) -> set[str]:
    """從所有投影片的 `slide_text` 抽出候選術語。

    取的是**標點／空白分隔的完整片段**，不是任意 n-gram。
    不做斷詞——斷詞需要詞典，而中文詞典的繁簡問題正是 D12 記的坑。

    **這個區別是實測逼出來的**（R20 §ngram_vs_unit）：

    | 詞庫 | 詞數 | 人工字幕上的誤報 | Whisper 上的 precision |
    |---|---|---|---|
    | 任意 n-gram (3–5) | 1190 | 19 | 91% |
    | **標點分隔單位** | **107** | **5** | **100%** |

    任意 n-gram 會把投影片某行的碎片（`以開始`、`任何意`、`生命只`）
    當成術語，於是把**正確的**逐字稿改壞：`一開始`→`以開始`。
    在錯誤基率高的 Whisper 逐字稿上這個問題被掩蓋，換到人工字幕就現形。
    """
    terms: set[str] = set()
    for slide in slides:
        for line in (getattr(slide, "slide_text", None) or "").splitlines():
            for run in _HAN_RUN.findall(line):
                if MIN_TERM_LEN <= len(run) <= MAX_TERM_LEN:
                    terms.add(run)
    return terms


def propose(text: str, lexicon: set[str], min_similarity: float) -> list[tuple[str, str, float]]:
    """在 `text` 中找出「拼音與某個術語相同、但寫法不同」的片段。

    只看**詞庫中有、但 `text` 裡沒有**的術語——已經寫對的不需要校正。

    回傳 `(錯的片段, 正確寫法, 拼音相似度)`，同一個片段只留最像的那筆。
    """
    from ..validation.corrections import pinyin_similarity

    flat = re.sub(r"\s+", "", text)
    if not flat:
        return []

    best: dict[str, tuple[str, str, float]] = {}
    for term in lexicon:
        if term in flat:
            continue
        n = len(term)
        for i in range(len(flat) - n + 1):
            candidate = flat[i : i + n]
            if candidate == term:
                continue
            score = pinyin_similarity(candidate, term)
            if score < min_similarity:
                continue
            if candidate not in best or score > best[candidate][2]:
                best[candidate] = (candidate, term, score)
    return sorted(best.values(), key=lambda row: -row[2])


def corrections_for(segment, lexicon: set[str], cfg) -> list:
    """為一個 segment 產生通過驗證的 `Correction`。

    **仍走 R13 的三道閘**（插入／語意改寫／大小寫），不繞過任何既有把關。
    §5.3 不變量 10（`from` 須出現在 `transcript_raw`）由 `propose` 的
    掃描方式天然滿足——候選就是從 `transcript_raw` 切出來的。
    """
    from ..ir import Correction, CorrectionMethod
    from ..validation.corrections import unauthorized_reason

    existing = {(c.from_text, c.to_text) for c in (segment.corrections or [])}
    out: list = []
    for from_text, to_text, score in propose(
        segment.transcript_raw, lexicon, cfg.min_pinyin_similarity
    ):
        if (from_text, to_text) in existing:
            continue
        reason = unauthorized_reason(from_text, to_text)
        if reason is not None:
            log.debug("%s：詞庫提案 %r→%r 未過 R13 的閘——%s",
                      segment.segment_id, from_text, to_text, reason)
            continue
        out.append(
            Correction(
                **{
                    "from": from_text,
                    "to": to_text,
                    "source": "lexicon",
                    "method": CorrectionMethod.LEXICON,
                    "reason": f"投影片上寫作「{to_text}」，逐字稿此處同音異寫",
                    "score": round(score, 3),
                }
            )
        )
    return out


def apply_to_video(segments, slides, transcript, cfg) -> int:
    """對整支影片跑一輪詞庫校正。回傳新增的校正筆數。

    **必須在 S4 全部跑完之後**——詞庫要從**所有**投影片建起來，
    第 5 張投影片上的術語可能在第 20 段被聽錯。
    """
    from .understand import apply_corrections

    if not cfg.enabled:
        return 0

    lexicon = build_lexicon(slides)
    if not lexicon:
        log.info("詞庫為空（沒有任何 slide_text），跳過事後校正")
        return 0

    added = 0
    for segment in segments:
        proposed = corrections_for(segment, lexicon, cfg)
        if not proposed:
            continue
        merged = list(segment.corrections or []) + proposed
        apply_corrections(transcript, segment, merged)
        added += len(proposed)
        for c in proposed:
            log.info("%s：詞庫校正 %r→%r（拼音 %.2f）",
                     segment.segment_id, c.from_text, c.to_text, c.score or 0.0)

    log.info("S4b %s：詞庫 %d 詞，新增 %d 筆校正",
             transcript.video_id, len(lexicon), added)
    return added
