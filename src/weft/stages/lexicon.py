"""術語詞庫萃取與逐字稿校正。SDD §4.4（詞庫）、§4.5（校正）。

§4.5 開宗明義：「這是本專案品質的關鍵路徑，不是可選功能。」
而它的門檻是 **precision ≥ 0.90**——寧可漏改，不可亂改。整個模組的設計
偏向保守：每一道判斷都在問「有沒有理由不改」，而不是「能不能改」。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: 書名號與引號內的文字必定是術語（SDD §4.4）
_BRACKETED = re.compile(r"[《〈「『【]([^》〉」』】]{1,16})[》〉」』】]")
#: 書名／篇名的長度上限。比一般術語寬鬆，理由見 extract_terms。
_BRACKET_MAX_LEN = 16
_CJK = re.compile(r"[一-鿿]+")
#: 中文標點與換行，用來切出「完整語句」
_CLAUSE_SPLIT = re.compile(r"[，。；：、！？\s\n．,.;:!?()（）《》〈〉「」『』【】]+")

#: 純功能字組成的片段不是術語。文言文虛詞尤其容易被分詞器切成雙字詞。
_FUNCTION_CHARS = set("的了是在有和與及也者其之以而則於為即乃亦所因故若")

#: OCR 的「短行」——標題與條列項——幾乎都是純術語，適合逐一取 2-gram。
#: 長行是完整句子，取 2-gram 只會產生跨詞邊界的碎片。
_SHORT_LINE_CHARS = 8

#: 常用詞的詞頻下限。超過此值的片段**不得被當作 ASR 錯字修改**。
#:
#: 實測（jieba 詞頻，繁體先轉簡再查）：
#:   常用詞  一個 142747、我們 98740、這個 61310、今天 15960、階段 9574、時機 1771
#:   ASR 錯字 時運 66、經血 50、羊神 0、形照 0
#:   領域術語 精血 23、識蘊 3、陽神 2、七魄 0、形兆 0
#: 300 落在 66 與 1771 之間（幾何中點約 342），兩側各留約 4 倍餘裕。
#: 見 docs/decisions.md D12。
_COMMON_WORD_FREQ = 300


def _is_plausible_term(term: str, min_len: int, max_len: int) -> bool:
    if not (min_len <= len(term) <= max_len):
        return False
    if not _CJK.fullmatch(term):
        return False
    # 全部由虛詞組成 → 不是術語
    return not all(c in _FUNCTION_CHARS for c in term)


def extract_terms(text: str, cfg) -> list[str]:
    """從一頁 OCR 文字抽出候選術語。SDD §4.4。

    三個來源：書名號／引號內文字、分詞結果中的多字詞、四字詞。
    """
    import jieba

    found: list[str] = []
    for match in _BRACKETED.finditer(text):
        inner = match.group(1).strip()
        # 書名不受 term_max_len 限制：《太上老君內觀經》有 7 字，本專案的
        # 核心經典就會被 6 字上限濾掉。書名號本身已經是明確的術語標記，
        # 不需要再用長度猜。
        if _is_plausible_term(inner, cfg.term_min_len, _BRACKET_MAX_LEN):
            found.append(inner)

    for token in jieba.cut(text):
        if _is_plausible_term(token, cfg.term_min_len, cfg.term_max_len):
            found.append(token)

    # 短行（標題、條列項）逐一取 2-gram。這是為了撈出分詞器切錯的文言術語：
    # jieba 的詞典是簡體，把「識蘊入胎次第」切成「蘊入」，SDD §3.3 親自舉例的
    # 「識蘊」因此永遠進不了詞庫。取 2-gram 會一併帶入「蘊入」這類碎片，
    # 但校正端有常用詞守門（見 _is_common_word），precision 不受影響。
    for line in text.splitlines():
        line = line.strip()
        if 2 <= len(line) <= _SHORT_LINE_CHARS and _CJK.fullmatch(line):
            for i in range(len(line) - 1):
                pair = line[i : i + 2]
                if _is_plausible_term(pair, 2, cfg.term_max_len):
                    found.append(pair)

    # 四字詞：取**以標點分隔的完整四字語句**，不是每個 4-gram。
    # 滑動視窗會產生「陽神為三」「靈為七魄」這類跨詞邊界的碎片，它們不是術語，
    # 卻會成為 §4.5 校正時的替換目標——直接傷害 precision（門檻 0.90）。
    for clause in _CLAUSE_SPLIT.split(text):
        clause = clause.strip()
        if len(clause) == 4 and _is_plausible_term(clause, 4, cfg.term_max_len):
            found.append(clause)

    return found


def build_lexicon(slides, series_id: str | None, cfg, existing=None):
    """累積系列級詞庫。SDD §4.4：**scope 為 series_id，不是單片**。"""
    from ..ir import Lexicon, LexiconEntry

    entries: dict[str, LexiconEntry] = {}
    if existing is not None:
        entries = {e.term: e.model_copy(deep=True) for e in existing.entries}

    for slide in slides:
        if not slide.ocr_text:
            continue
        for term in extract_terms(slide.ocr_text, cfg):
            entry = entries.get(term)
            if entry is None:
                entries[term] = LexiconEntry(term=term, pinyin=pinyin_key(term), count=1)
                entry = entries[term]
            else:
                entry.count += 1
            # first_seen 記錄該術語出現在哪支影片的哪幾張投影片，供 §4.5
            # 的「只在時間鄰近的投影片詞庫中比對」使用
            seen = entry.first_seen.setdefault(slide.slide_id.split("#")[0], [])
            if slide.slide_id not in seen:
                seen.append(slide.slide_id)

    kept = [e for e in entries.values() if e.count >= cfg.min_count]
    return Lexicon(
        series_id=series_id,
        entries=sorted(kept, key=lambda e: (-e.count, e.term)),
        ocr_model=existing.ocr_model if existing else None,
    )


# --------------------------------------------------------------------------
# 相似度
# --------------------------------------------------------------------------


def is_common_word(fragment: str) -> bool:
    """這個片段本身是不是常用詞？是的話**不得**當作 ASR 錯字修改。

    這是 precision 的主要守門。jieba 的詞典是簡體，所以先繁轉簡再查——
    不轉的話「我們」「這個」的詞頻都是 0，守門形同虛設。
    """
    import jieba
    from zhconv import convert

    jieba.initialize()  # FREQ 在初始化前是空的
    return jieba.dt.FREQ.get(convert(fragment, "zh-cn"), 0) >= _COMMON_WORD_FREQ


def pinyin_key(text: str) -> str:
    """帶聲調的拼音序列，以空白分隔。

    **保留聲調**：不保留的話同音字範圍會擴大數倍，誤改的機會跟著放大，
    而 precision 是這個階段的硬門檻。
    """
    from pypinyin import Style, lazy_pinyin

    return " ".join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))


def _sequence_similarity(a: list[str], b: list[str]) -> float:
    """兩個音節序列的編輯相似度。"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != bj))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def pinyin_similarity(a: str, b: str) -> float:
    return _sequence_similarity(pinyin_key(a).split(), pinyin_key(b).split())


def shape_similarity(a: str, b: str) -> float:
    """共用字元的比例。ASR 常只錯其中一個字，此時字形仍有部分重疊。"""
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa), len(sb)) if sa and sb else 0.0


# --------------------------------------------------------------------------
# 校正
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    term: str
    slide_id: str
    score: float


def _score(fragment: str, term: str, cfg) -> float:
    """拼音為主訊號，字形只能加分、不能扣分。

    中文 ASR 的錯誤絕大多數是同音或近音（SDD §4.5 原文即如此描述），
    所以拼音一致本身就是強證據。若把字形當作等權的第二項，「時運 → 識蘊」
    這種**零字形重疊的正確修正**會被壓到門檻以下——那正是 SDD §3.3 舉的
    例子，把它擋掉等於這個階段沒做事。
    """
    pin = pinyin_similarity(fragment, term)
    shape = shape_similarity(fragment, term)
    return pin + (1.0 - pin) * cfg.shape_weight * shape


def correct_transcript(transcript, lexicon, slide_windows, cfg):
    """就地校正逐字稿。回傳 `(transcript, corrections)`。

    `slide_windows` 為 `[(t_start, t_end, slide_id)]`，用來限制「只在時間上
    鄰近的投影片詞庫中比對」（§4.5 約束 2）——避免用第 40 分鐘的術語去改
    第 5 分鐘的話。

    **`text_raw` 永不覆寫**（§4.5 約束 3、§5.3 不變量 9）。
    """
    from ..ir import Correction

    if lexicon is None or not lexicon.entries:
        log.info("詞庫為空，跳過 S2c（§4.5 失敗行為）")
        for cue in transcript.cues:
            cue.text_corrected = cue.text_raw
            cue.corrections = []
        return transcript, []

    by_slide: dict[str, list] = {}
    for entry in lexicon.entries:
        for slide_ids in entry.first_seen.values():
            for slide_id in slide_ids:
                by_slide.setdefault(slide_id, []).append(entry)

    ordered_slides = [s for _, _, s in sorted(slide_windows)]
    slide_position = {s: i for i, s in enumerate(ordered_slides)}
    known_terms = {e.term for e in lexicon.entries}

    all_corrections: list[Correction] = []
    for cue in transcript.cues:
        nearby = _nearby_entries(
            cue, slide_windows, slide_position, ordered_slides, by_slide, cfg.neighbor_window
        )
        text, corrections = _correct_one(cue.text_raw, nearby, known_terms, cfg)
        cue.text_corrected = text
        cue.corrections = corrections
        all_corrections += corrections

    log.info("S2c：套用 %d 筆術語校正", len(all_corrections))
    return transcript, all_corrections


def _nearby_entries(
    cue, slide_windows, slide_position, ordered_slides, by_slide, window: int
) -> list[tuple[object, str]]:
    """取時間上鄰近的投影片所貢獻的詞庫條目。§4.5 約束 2。"""
    centre = None
    for t_start, t_end, slide_id in slide_windows:
        if t_start <= cue.t_start < t_end:
            centre = slide_position[slide_id]
            break
    if centre is None:
        # 純講者時段：取時間最近的投影片作為中心
        if not slide_windows:
            return []
        nearest = min(slide_windows, key=lambda w: min(abs(cue.t_start - w[0]), abs(cue.t_start - w[1])))
        centre = slide_position[nearest[2]]

    lo, hi = max(0, centre - window), min(len(ordered_slides), centre + window + 1)
    out: list[tuple[object, str]] = []
    for slide_id in ordered_slides[lo:hi]:
        for entry in by_slide.get(slide_id, []):
            out.append((entry, slide_id))
    return out


def _correct_one(text: str, nearby, known_terms: set[str], cfg):
    """在單句中找出並替換 ASR 錯字。

    由左至右掃描，同一個位置只改一次；長片段優先，避免把「識蘊」拆成
    兩次單字替換。
    """
    from ..ir import Correction

    corrections: list[Correction] = []
    result: list[str] = []
    i = 0
    lengths = sorted({len(e.term) for e, _ in nearby}, reverse=True)

    while i < len(text):
        matched = False
        for length in lengths:
            fragment = text[i : i + length]
            if len(fragment) < 2 or not _CJK.fullmatch(fragment):
                continue
            # 片段本身已是詞庫條目 → 它是對的，不動
            if fragment in known_terms:
                continue
            # 片段是常用詞 → 講者本來就在講這個詞，不是 ASR 錯字。
            # 這是 precision 的主要守門（§5.2 門檻 0.90）。
            if is_common_word(fragment):
                continue

            best: Candidate | None = None
            for entry, slide_id in nearby:
                if len(entry.term) != length or entry.term == fragment:
                    continue
                score = _score(fragment, entry.term, cfg)
                if score >= cfg.similarity_threshold and (best is None or score > best.score):
                    best = Candidate(entry.term, slide_id, score)

            if best is not None:
                result.append(best.term)
                corrections.append(
                    Correction(
                        **{
                            "from": fragment,
                            "to": best.term,
                            "source": best.slide_id,
                            "method": "lexicon",
                            "score": round(best.score, 4),
                        }
                    )
                )
                i += length
                matched = True
                break
        if not matched:
            result.append(text[i])
            i += 1

    return "".join(result), corrections
