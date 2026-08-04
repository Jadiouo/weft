"""術語詞庫與逐字稿校正。SDD §4.4、§4.5。

§4.5：「這是本專案品質的關鍵路徑，不是可選功能。」
§5.2：術語校正 **precision ≥ 0.90**——寧可漏改，不可亂改。

precision 的門檻測試在此以**同音錯誤注入**的方式做：取一段已知正確的文字，
注入真實 ASR 會犯的同音錯誤，然後量測修回率與誤改率。這與 §5.1（B）的
真實影片黃金集互補——黃金集測的是真實 Whisper 的錯誤分布，這裡測的是
演算法在已知錯誤下的行為，成本低且可重現。
"""

from __future__ import annotations

import pytest

from weft.config import S2bConfig, S2cConfig
from weft.ir import Lexicon, LexiconEntry, Transcript, TranscriptCue, TranscriptSource
from weft.stages.lexicon import (
    build_lexicon,
    correct_transcript,
    extract_terms,
    pinyin_key,
    pinyin_similarity,
    shape_similarity,
)
from weft.validation.metrics import correction_prf
from weft.validation.thresholds import TERM_CORRECTION_PRECISION

SLIDE_TEXT = (
    "一月為胞，精血凝也。二月為胎，形兆胚也。"
    "三月陽神為三魂，動而生也。四月陰靈為七魄，靜鎮形也。《太上老君內觀經》"
)


def cfg2b() -> S2bConfig:
    return S2bConfig()


def cfg2c() -> S2cConfig:
    return S2cConfig()


# --------------------------------------------------------------------------
# 詞庫萃取（§4.4）
# --------------------------------------------------------------------------


def test_extracts_domain_terms():
    terms = set(extract_terms(SLIDE_TEXT, cfg2b()))
    for expected in ("精血", "陽神", "三魂", "七魄", "形兆"):
        assert expected in terms, f"漏抽術語 {expected}"


def test_extracts_bracketed_text():
    """SDD §4.4：書名號與括號內文字。"""
    assert "太上老君內觀經" in set(extract_terms("講的是《太上老君內觀經》這一部", cfg2b()))


def test_extracts_four_character_clauses():
    """SDD §4.4 的「四字詞」。"""
    terms = set(extract_terms(SLIDE_TEXT, cfg2b()))
    assert "精血凝也" in terms
    assert "形兆胚也" in terms


def test_does_not_extract_sliding_window_fragments():
    """滑動視窗會產生「陽神為三」「靈為七魄」這類跨詞邊界的碎片。
    它們不是術語，卻會成為校正時的替換目標——直接傷害 precision。"""
    terms = set(extract_terms(SLIDE_TEXT, cfg2b()))
    for junk in ("陽神為三", "靈為七魄", "月陰靈為", "上老君內"):
        assert junk not in terms, f"抽到跨詞邊界的碎片 {junk}"


def test_pure_function_words_are_not_terms():
    assert "而其" not in set(extract_terms("而其之於也者", cfg2b()))


def test_lexicon_scope_is_series_level():
    """SDD §4.4：詞庫 scope 為 series_id，**不是單片級**——
    同系列後續影片可直接受惠。"""
    from weft.ir import Slide

    slides = [Slide(slide_id="slide_001", image_path="a.png", t_first_seen=0, t_last_seen=1,
                    ocr_text=SLIDE_TEXT)]
    lex = build_lexicon(slides, "PL_series", cfg2b())
    assert lex.series_id == "PL_series"


def test_lexicon_accumulates_across_videos():
    """新影片的術語 append 進去，既有的計數累加。"""
    from weft.ir import Slide

    first = build_lexicon(
        [Slide(slide_id="slide_001", image_path="a.png", t_first_seen=0, t_last_seen=1,
               ocr_text="精血凝也")],
        "PL", cfg2b(),
    )
    second = build_lexicon(
        [Slide(slide_id="slide_002", image_path="b.png", t_first_seen=0, t_last_seen=1,
               ocr_text="陽神為三魂")],
        "PL", cfg2b(), existing=first,
    )
    terms = {e.term for e in second.entries}
    assert "精血" in terms and "陽神" in terms


# --------------------------------------------------------------------------
# 相似度
# --------------------------------------------------------------------------


def test_pinyin_keeps_tones():
    """不保留聲調會讓同音字範圍擴大數倍，誤改機會跟著放大——
    而 precision 是這階段的硬門檻。"""
    assert pinyin_key("識") != pinyin_key("是")


def test_homophone_pairs_score_high():
    """SDD §3.3 的示例：時運 → 識蘊。"""
    assert pinyin_similarity("時運", "識蘊") == 1.0


def test_shape_similarity_is_partial_overlap():
    assert shape_similarity("經血", "精血") == 0.5
    assert shape_similarity("時運", "識蘊") == 0.0


def test_shape_never_penalises_a_correct_homophone():
    """關鍵：「時運 → 識蘊」的字形重疊是 **0**。若把字形當等權第二項，
    這個 SDD 親自舉的正確修正會被壓到門檻以下，等於這階段沒做事。"""
    from weft.stages.lexicon import _score

    assert _score("時運", "識蘊", cfg2c()) >= cfg2c().similarity_threshold


# --------------------------------------------------------------------------
# 校正行為（§4.5 的四項約束）
# --------------------------------------------------------------------------


def make_transcript(rows: list[tuple[float, float, str]]) -> Transcript:
    cues = [TranscriptCue(index=i, t_start=a, t_end=b, text_raw=t) for i, (a, b, t) in enumerate(rows)]
    return Transcript(
        video_id="v", source=TranscriptSource.WHISPER, cues=cues,
        raw_hash=Transcript.compute_raw_hash(cues),
    )


def make_lexicon(terms: dict[str, str]) -> Lexicon:
    """`{術語: slide_id}`。"""
    return Lexicon(
        series_id="PL",
        entries=[
            LexiconEntry(term=t, pinyin=pinyin_key(t), count=3, first_seen={"v": [s]})
            for t, s in terms.items()
        ],
    )


def test_homophone_error_is_corrected():
    transcript = make_transcript([(0.0, 10.0, "這裡的時運其實是入胎的時機")])
    lexicon = make_lexicon({"識蘊": "slide_001"})
    result, corrections = correct_transcript(
        transcript, lexicon, [(0.0, 20.0, "slide_001")], cfg2c()
    )
    assert "識蘊" in result.cues[0].text_corrected
    assert len(corrections) == 1
    assert (corrections[0].from_text, corrections[0].to_text) == ("時運", "識蘊")


def test_raw_text_is_never_overwritten():
    """§4.5 約束 3、§5.3 不變量 9。"""
    transcript = make_transcript([(0.0, 10.0, "這裡的時運其實是入胎的時機")])
    before = transcript.raw_hash
    result, _ = correct_transcript(
        transcript, make_lexicon({"識蘊": "slide_001"}), [(0.0, 20.0, "slide_001")], cfg2c()
    )
    assert result.cues[0].text_raw == "這裡的時運其實是入胎的時機"
    assert Transcript.compute_raw_hash(result.cues) == before
    assert result.raw_is_intact()


def test_every_correction_is_recorded():
    """§4.5 約束 1：每次替換必須記錄 {from, to, source, method, score}。"""
    transcript = make_transcript([(0.0, 10.0, "時運與經血都要改")])
    _, corrections = correct_transcript(
        transcript,
        make_lexicon({"識蘊": "slide_001", "精血": "slide_001"}),
        [(0.0, 20.0, "slide_001")],
        cfg2c(),
    )
    assert len(corrections) == 2
    for c in corrections:
        assert c.source.startswith("slide_")
        assert c.method == "lexicon"
        assert 0.0 <= c.score <= 1.0


def test_corrections_are_traceable_to_raw_text():
    """§5.3 不變量 10：每筆的 from 字串必須實際出現在 transcript_raw 中。"""
    transcript = make_transcript([(0.0, 10.0, "這裡的時運其實是入胎的時機")])
    result, corrections = correct_transcript(
        transcript, make_lexicon({"識蘊": "slide_001"}), [(0.0, 20.0, "slide_001")], cfg2c()
    )
    for c in corrections:
        assert c.from_text in result.cues[0].text_raw


def test_only_temporally_nearby_terms_are_used():
    """§4.5 約束 2：不得用第 40 分鐘的術語去改第 5 分鐘的話。"""
    transcript = make_transcript([(0.0, 10.0, "這裡的時運其實是入胎的時機")])
    lexicon = make_lexicon({"識蘊": "slide_020"})  # 出現在很後面的投影片
    windows = [(i * 100.0, (i + 1) * 100.0, f"slide_{i + 1:03d}") for i in range(20)]

    _, corrections = correct_transcript(transcript, lexicon, windows, cfg2c())
    assert corrections == [], "用了時間上不相鄰的投影片術語"


def test_empty_lexicon_skips_cleanly():
    """§4.5 失敗行為：詞庫為空 → 跳過，corrected = raw。"""
    transcript = make_transcript([(0.0, 10.0, "這裡的時運其實是入胎的時機")])
    result, corrections = correct_transcript(transcript, Lexicon(), [], cfg2c())
    assert corrections == []
    assert result.cues[0].text_corrected == result.cues[0].text_raw


def test_already_correct_term_is_left_alone():
    """逐字稿已經寫對的術語不該被改（即使有同音的其他詞庫條目）。"""
    transcript = make_transcript([(0.0, 10.0, "這裡的識蘊其實是入胎的時機")])
    _, corrections = correct_transcript(
        transcript, make_lexicon({"識蘊": "slide_001"}), [(0.0, 20.0, "slide_001")], cfg2c()
    )
    assert corrections == []


# --------------------------------------------------------------------------
# §5.2 precision 門檻
# --------------------------------------------------------------------------

#: 真實中文 ASR 會犯的同音／近音錯誤。左為錯字，右為正解。
_INJECTED = [
    ("時運", "識蘊"),
    ("經血", "精血"),
    ("羊神", "陽神"),
    ("形照", "形兆"),
    ("內光經", "內觀經"),
]

#: 不含任何術語的句子。校正器在這些句子上**不該動任何東西**——
#: 誤改一處就直接扣 precision。
_CLEAN_SENTENCES = [
    "今天我們接著看下面這一段",
    "這個地方大家要特別注意",
    "我先講一個比較容易懂的比喻",
    "所以說這件事情並不是那麼簡單",
    "各位如果有問題可以隨時提出來",
    "剛才提到的那個部分我再說明一次",
    "我們下個禮拜再繼續講",
    "這樣講大家應該就清楚了",
]


def test_correction_precision_meets_threshold():
    """§5.2：術語校正 precision ≥ 0.90。

    §5.5 #7：**不得為了讓測試通過而調低此門檻。**
    §5.5 #11：這裡是量化門檻，不是「跑過就好」的斷言。
    """
    rows: list[tuple[float, float, str]] = []
    expected: list[tuple[int, str, str]] = []

    # 注入已知錯誤的句子
    for i, (wrong, right) in enumerate(_INJECTED):
        rows.append((i * 10.0, i * 10.0 + 9.0, f"這一段講的是{wrong}的意思"))
        expected.append((i, wrong, right))

    # 乾淨句子：任何改動都是誤改
    offset = len(_INJECTED)
    for j, sentence in enumerate(_CLEAN_SENTENCES):
        rows.append(((offset + j) * 10.0, (offset + j) * 10.0 + 9.0, sentence))

    transcript = make_transcript(rows)
    lexicon = make_lexicon({right: "slide_001" for _, right in _INJECTED})
    windows = [(0.0, rows[-1][1] + 10.0, "slide_001")]

    result, _ = correct_transcript(transcript, lexicon, windows, cfg2c())

    applied = [
        (cue.index, c.from_text, c.to_text)
        for cue in result.cues
        for c in cue.corrections
    ]
    prf = correction_prf(applied, expected)

    assert prf.precision >= TERM_CORRECTION_PRECISION, (
        f"precision={prf.precision:.3f} < {TERM_CORRECTION_PRECISION}\n"
        f"  誤改：{sorted(set(applied) - set(expected))}"
    )
    # recall 依 §5.2「記錄但不設硬門檻」，只報告
    print(f"\n術語校正：{prf}（recall 記錄用，不設門檻）")


def test_clean_sentences_are_untouched():
    """precision 的另一面：沒有術語的句子必須原封不動。"""
    rows = [(i * 10.0, i * 10.0 + 9.0, s) for i, s in enumerate(_CLEAN_SENTENCES)]
    transcript = make_transcript(rows)
    lexicon = make_lexicon({right: "slide_001" for _, right in _INJECTED})

    result, corrections = correct_transcript(
        transcript, lexicon, [(0.0, 1000.0, "slide_001")], cfg2c()
    )
    assert corrections == [], f"在乾淨句子上誤改：{[(c.from_text, c.to_text) for c in corrections]}"
    for cue in result.cues:
        assert cue.text_corrected == cue.text_raw
