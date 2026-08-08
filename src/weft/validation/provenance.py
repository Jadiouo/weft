"""SDD §5.4 溯源檢查 —— 防幻覺閘門。

三道檢查：
  正向：block 內容能否在來源中找到（找不到 → unverified）
  反向：block 內容是否只是逐字複製（複製過頭 → degenerate_copy）
  具名實體：人名／書名／數字／年代必須在來源出現

相似度刻意用**字元 n-gram**而非 embedding：溯源檢查要回答的是「這句話的
材料是否來自來源」，不是「語意上像不像」。embedding 會把「講者說了 A，
LLM 寫成語意相近但事實不同的 B」判為相似，那正是要抓的東西。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import ProvenanceConfig
from ..ir import ContentBlock, ProvenanceKind, VerificationStatus, VideoIR
from .thresholds import MAX_UNVERIFIED_RATIO

# --------------------------------------------------------------------------
# 相似度
# --------------------------------------------------------------------------


def _ngrams(text: str, n: int) -> set[str]:
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


#: 中文的詞多為一至二字，bigram 因此接近「詞」的粒度。
#:
#: n 的選擇是實測的結果，不是隨手挑的（實驗見 docs/decisions.md）：
#:   n=1 太寬鬆——「的、是、為、也」這類高頻字會把域內幻覺灌到 0.36–0.48
#:   n=3 太嚴苛——忠實改寫必然重組詞序，3-gram 在每個接縫斷掉，掉到 0.23
#:   n=2 分離最乾淨——忠實改寫 ≥0.364，重用術語的對抗性幻覺 ≤0.154（2.4 倍）
DEFAULT_NGRAM = 2


def containment(candidate: str, source: str, n: int = DEFAULT_NGRAM) -> float:
    """candidate 的 n-gram 有多少比例出現在 source 中。

    用 containment 而非 Jaccard：來源（整頁 OCR 或整段逐字稿）通常遠長於
    單一 block，Jaccard 會被長度差稀釋成永遠很低。
    """
    cand = _ngrams(candidate, n)
    if not cand:
        return 0.0
    src = _ngrams(source, n)
    return len(cand & src) / len(cand)


def longest_common_substring_ratio(candidate: str, source: str) -> float:
    """最長共同子串長度 ÷ candidate 長度。用於反向檢查的逐字複製率。

    比 n-gram containment 更直接：一段被整句搬過來的文字，LCS 會等於它
    自己的長度；而一段真正經過改寫整合的文字，即使用詞高度重疊，也不會有
    很長的連續共同片段。
    """
    a = re.sub(r"\s+", "", candidate)
    b = re.sub(r"\s+", "", source)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best / len(a)


# --------------------------------------------------------------------------
# 具名實體
# --------------------------------------------------------------------------

_BOOK_TITLE = re.compile(r"《([^》]{1,30})》")
_ARABIC_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CJK_DIGITS = "零一二三四五六七八九十百千萬億兩"
#: 數字後面的單位。**清單不完整就等於漏抓**——原本只有
#: 「年月日歲個張條章卷篇位次倍分」，漏掉週／天／種／成／公分等常見單位，
#: 導致「分為四種」「佔七成」「第五週」「二十一天」全部抓不到。
#: R12 校準實測：補上後，口頭延伸的幻覺攔截率從 0/5 升到 3/5，**誤報零增加**。
#:
#: 多字單位放前面（`公分` 必須排在 `分` 之前），否則會被單字單位先吃掉。
_CJK_UNITS = (
    "公分|公尺|公里|公斤|公克|毫升|毫米|世紀|"
    "年|月|日|時|分|秒|週|周|天|旬|季|"
    "個|張|條|章|卷|篇|位|次|倍|種|類|項|點|步|層|級|成|度|歲|%|％"
)
_CJK_NUMBER = re.compile(rf"[{_CJK_DIGITS}]{{1,8}}(?=(?:{_CJK_UNITS}))")
_YEAR = re.compile(rf"(?:\d{{2,4}}|[{_CJK_DIGITS}]{{1,6}})年")


@dataclass(frozen=True)
class NamedEntities:
    """§5.4 的「人名、書名、數字、年代」。

    **人名不做自動抽取。** 中文人名無邊界標記，任何 regex 啟發式都會把
    「太上老君」「陰陽和合」之類的詞誤判成人名，製造大量假 unverified，
    反而讓真正的幻覺淹沒在雜訊裡。人名的溯源改由 S2b 的系列術語詞庫涵蓋
    （專有名詞本來就會進詞庫），這是 Phase 2 的工作。此限制刻意留在型別
    定義上，不用空欄位假裝已完成。
    """

    book_titles: frozenset[str] = frozenset()
    numbers: frozenset[str] = frozenset()
    years: frozenset[str] = frozenset()

    def all(self) -> frozenset[str]:
        return self.book_titles | self.numbers | self.years


def extract_named_entities(text: str) -> NamedEntities:
    return NamedEntities(
        book_titles=frozenset(_BOOK_TITLE.findall(text)),
        numbers=frozenset(_ARABIC_NUMBER.findall(text)) | frozenset(_CJK_NUMBER.findall(text)),
        years=frozenset(_YEAR.findall(text)),
    )


_CJK_VALUE = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
_ARABIC_TO_CJK = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
                  "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}


def number_variants(entity: str) -> set[str]:
    """同一個數值的其他寫法。

    實測：block 寫「**兩**組靜脈」，投影片寫「**2**組靜脈」——同一個數字，
    精確比對抓不到，於是整個 block 被判未通過（R27）。
    「兩」與「二」也是同一個值。

    **只處理十以下的單字數字**。十以上的中文數字（十五、二十四）換算規則
    複雜，而且真正的幻覺多半改的是量級不是寫法；為了少數寫法差異去寫一個
    半對的換算器，反而會讓「三十」與「三」互相匹配那種錯誤放行。
    """
    out = {entity}
    if len(entity) == 1:
        if entity in _CJK_VALUE:
            v = _CJK_VALUE[entity]
            out.add(str(v))
            out |= {c for c, n in _CJK_VALUE.items() if n == v}
        elif entity in _ARABIC_TO_CJK:
            out.add(_ARABIC_TO_CJK[entity])
            if entity == "2":
                out.add("兩")
    return out


def unsupported_entities(candidate: str, source: str) -> list[str]:
    """candidate 中出現、但來源沒有的具名實體。

    數字比對前先正規化寫法（`number_variants`）——「兩」與「2」是同一個數，
    寫法不同不是編造。書名與年代仍是精確比對。
    """
    src = re.sub(r"\s+", "", source)
    return sorted(e for e in extract_named_entities(candidate).all()
                  if e and not any(w in src for w in number_variants(e)))


# --------------------------------------------------------------------------
# 檢查
# --------------------------------------------------------------------------


@dataclass
class BlockVerdict:
    segment_id: str
    block_index: int
    content_type: str
    status: VerificationStatus
    similarity: float
    copy_ratio: float
    missing_entities: list[str] = field(default_factory=list)
    reason: str = ""
    #: 未通過，但內容其實溯得到**同一段的另一個來源**（見 `_diagnose_wrong_source`）。
    #: 這是歸屬錯誤不是幻覺，**修法完全不同**——要修的是 S4c 的 prompt，
    #: 不是內容品質。仍然算未通過。
    wrong_source: bool = False


@dataclass
class VideoVerdict:
    video_id: str
    verdicts: list[BlockVerdict]

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def unverified(self) -> list[BlockVerdict]:
        return [v for v in self.verdicts if v.status is not VerificationStatus.VERIFIED]

    @property
    def wrong_source(self) -> list[BlockVerdict]:
        """未通過、但溯得到同段另一個來源的。修的是歸屬不是內容。"""
        return [v for v in self.verdicts if v.wrong_source]

    @property
    def unverified_ratio(self) -> float:
        return len(self.unverified) / self.total if self.total else 0.0

    @property
    def pass_rate(self) -> float:
        return 1.0 - self.unverified_ratio

    @property
    def needs_review(self) -> bool:
        """§5.4：unverified 比例 > 5% → 整支標記 needs_review，不進 chunks.jsonl。"""
        return self.unverified_ratio > MAX_UNVERIFIED_RATIO


def resolve_source(ir: VideoIR, block: ContentBlock, segment_transcript: str) -> str:
    """取出 provenance 指向的來源文字。"""
    if block.provenance.kind is ProvenanceKind.SLIDE_OCR:
        slide = ir.slide_by_id(block.provenance.ref)
        # v0.3：來源是 **VLM 讀出的**投影片文字（原本是獨立的本地 OCR）。
        # 這削弱了本檢查的獨立性——見 known-risks R9。prompt 以「先逐字
        # 轉錄、再詮釋」的欄位順序保留部分獨立性。
        return slide.slide_text or "" if slide else ""
    return segment_transcript


def check_block(
    block: ContentBlock,
    source: str,
    cfg: ProvenanceConfig,
    segment_id: str,
    block_index: int,
) -> BlockVerdict:
    sim = containment(block.text, source)
    copy_ratio = longest_common_substring_ratio(block.text, source)
    missing = unsupported_entities(block.text, source) if cfg.check_named_entities else []

    status = VerificationStatus.VERIFIED
    reason = ""

    # 來源長度比：擋「來源太短，撐不起這段內容」。containment 對多數型別
    # 已停用（見 config 的說明），這是正向檢查僅存的實質作用。
    source_ratio = (
        len(re.sub(r"\s+", "", source)) / len(re.sub(r"\s+", "", block.text))
        if block.text.strip() else 0.0
    )

    threshold = cfg.min_similarity_by_type.get(str(block.type), cfg.min_similarity)
    if source_ratio < cfg.min_source_ratio:
        status = VerificationStatus.UNVERIFIED
        reason = (f"來源長度僅 block 的 {source_ratio:.0%}"
                  f"（下限 {cfg.min_source_ratio:.0%}），撐不起這段內容")
    elif sim < threshold:
        status = VerificationStatus.UNVERIFIED
        reason = f"與來源相似度 {sim:.3f} < {threshold}（{block.type.value}）"
    elif missing:
        status = VerificationStatus.UNVERIFIED
        reason = f"具名實體未出現於來源：{missing}"
    elif copy_ratio >= cfg.copy_similarity:
        # 逐字複製本身不必然是錯——經文原文就該是引文。是否算失敗由型別的
        # max_copy_ratio 在影片層級判定，這裡只先標記。
        status = VerificationStatus.DEGENERATE_COPY
        reason = f"逐字複製率 {copy_ratio:.3f} ≥ {cfg.copy_similarity}"

    return BlockVerdict(
        segment_id=segment_id,
        block_index=block_index,
        content_type=str(block.type),
        status=status,
        similarity=sim,
        copy_ratio=copy_ratio,
        missing_entities=missing,
        reason=reason,
    )


def _diagnose_wrong_source(ir, seg, block, transcript: str, cfg, verdict) -> None:
    """未通過時，看看它是不是**溯得到同一段的另一個來源**，只是型別標錯。

    **只改 `reason`，不改 `status`。** 這一段有兩個來源（投影片文字與逐字稿），
    內容忠實於其中一個卻標成另一個，是**歸屬錯誤**而不是幻覺——
    但它仍然不算通過：§3.5 要求 chunk 自足，「投影片寫了 X」與
    「講者說了 X」在下游是兩件事，metadata 帶錯就是帶錯。

    存在的理由是**這兩種要分開修**。C1 實測四支影片 27 筆未通過：
      5 筆　來源投影片其實不是投影片（分類誤報的下游後果）
      8 筆　內容抄自同段投影片卻標成 transcript ← 這一類
     14 筆　兩個來源都對不上，才是真正要判「內容有沒有問題」的
    混在一個數字裡的時候，看起來像「內容品質不到 0.95」，
    其實有一半是別的地方壞掉。
    """
    if verdict.status is not VerificationStatus.UNVERIFIED:
        return
    if block.provenance.kind is ProvenanceKind.SLIDE_OCR:
        other, name = transcript, "逐字稿"
    else:
        slide = ir.slide_by_id(seg.slide_ref) if seg.slide_ref else None
        other, name = (slide.slide_text or "" if slide else ""), "同段的投影片文字"
    if not other.strip():
        return
    sim = containment(block.text, other)
    threshold = cfg.min_similarity_by_type.get(str(block.type), cfg.min_similarity)
    if sim >= threshold and sim > verdict.similarity * 2:
        verdict.reason += (f"；但它與**{name}**的相似度是 {sim:.3f}"
                           f"——來源型別可能標錯，不是內容編造")
        verdict.wrong_source = True


def check_video(ir: VideoIR, cfg: ProvenanceConfig) -> VideoVerdict:
    """跑完整支影片的溯源檢查，並就地填回 block 的 verification / similarity。"""
    verdicts: list[BlockVerdict] = []
    copy_counts: dict[str, list[int]] = {}

    for seg in ir.segments:
        if seg.understanding is None:
            continue
        transcript = seg.transcript_corrected or seg.transcript_raw
        for i, block in enumerate(seg.understanding.content_blocks):
            source = resolve_source(ir, block, transcript)
            verdict = check_block(block, source, cfg, seg.segment_id, i)
            _diagnose_wrong_source(ir, seg, block, transcript, cfg, verdict)
            block.verification = verdict.status
            block.similarity = verdict.similarity
            verdicts.append(verdict)
            bucket = copy_counts.setdefault(str(block.type), [0, 0])
            bucket[1] += 1
            if verdict.copy_ratio >= cfg.copy_similarity:
                bucket[0] += 1

    # 反向檢查的型別級判定：某型別若高複製率 block 的比例超過上限，
    # 該型別下所有高複製 block 才真的記為 degenerate_copy；否則放行。
    for verdict in verdicts:
        if verdict.status is not VerificationStatus.DEGENERATE_COPY:
            continue
        copied, total = copy_counts[verdict.content_type]
        limit = cfg.max_copy_ratio.get(verdict.content_type, 0.8)
        if total and copied / total <= limit:
            verdict.status = VerificationStatus.VERIFIED
            verdict.reason = ""

    for seg in ir.segments:
        if seg.understanding is None:
            continue
        by_index = {(v.segment_id, v.block_index): v for v in verdicts}
        for i, block in enumerate(seg.understanding.content_blocks):
            v = by_index.get((seg.segment_id, i))
            if v is not None:
                block.verification = v.status

    result = VideoVerdict(video_id=ir.meta.video_id, verdicts=verdicts)
    ir.unverified_ratio = result.unverified_ratio
    ir.needs_review = result.needs_review
    return result
