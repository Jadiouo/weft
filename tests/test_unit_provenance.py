"""SDD §5.4 溯源檢查的測試。

「這是防止 LLM 編造內容的唯一有效機制，必須實作。」

因此這裡的反例必須是**真的幻覺長什麼樣**：流暢、合理、與主題相符，但來源
裡沒有。一個只測「空字串會被擋」的檢查器毫無用處。
"""

from __future__ import annotations

import pytest

from weft.config import ProvenanceConfig
from weft.ir import (
    ContentBlock,
    ContentType,
    Provenance,
    ProvenanceKind,
    VerificationStatus,
)
from weft.validation import provenance as p
from weft.validation.thresholds import MAX_UNVERIFIED_RATIO

SLIDE = (
    "一月為胞，精血凝也。二月為胎，形兆胚也。"
    "三月陽神為三魂，動而生也。四月陰靈為七魄，靜鎮形也。"
)
TRANSCRIPT = (
    "我們看這一段，一月為胞講的是受精卵凝聚成形的階段，"
    "父精母血在此時凝結。講者用簽約來比喻識蘊進入的時機。"
)


def cfg() -> ProvenanceConfig:
    return ProvenanceConfig()


def block(text: str, ctype: ContentType = ContentType.VERNACULAR) -> ContentBlock:
    return ContentBlock(
        type=ctype,
        text=text,
        provenance=Provenance(kind=ProvenanceKind.SLIDE_OCR, ref="slide_001"),
    )


# --------------------------------------------------------------------------
# 相似度基本性質
# --------------------------------------------------------------------------


def test_containment_of_exact_quote_is_one():
    assert p.containment("一月為胞，精血凝也。", SLIDE) == 1.0


def test_containment_ignores_source_length():
    """來源比 block 長很多是常態（整頁 OCR vs 一句話）。用 Jaccard 的話
    這裡會被稀釋成很低的分數，整個檢查就失效。"""
    short = "一月為胞，精血凝也。"
    assert p.containment(short, SLIDE) == 1.0
    assert p.containment(short, SLIDE * 20) == 1.0


def test_containment_of_unrelated_text_is_low():
    assert p.containment("量子力學的測不準原理說明了觀測者效應", SLIDE) < 0.2


def test_containment_ignores_whitespace():
    assert p.containment("一月為胞，\n精血凝也。", SLIDE) == 1.0


def test_containment_of_empty_candidate_is_zero():
    assert p.containment("", SLIDE) == 0.0


# --------------------------------------------------------------------------
# 正向檢查：幻覺必須被抓到
# --------------------------------------------------------------------------


def test_verbatim_scripture_passes():
    v = p.check_block(block("一月為胞，精血凝也。", ContentType.SCRIPTURE), SLIDE, cfg(), "s#1", 0)
    assert v.status is not VerificationStatus.UNVERIFIED


def test_faithful_paraphrase_passes():
    """真正的白話解說：用詞大量取自來源，只是重組。這種必須放行，
    否則 S4 會被逼成只能複製貼上——那正是反向檢查要擋的東西。"""
    text = "一月為胞，是指精血凝聚；二月為胎，形兆已成胚。"
    v = p.check_block(block(text), SLIDE, cfg(), "s#1", 0)
    assert v.status is VerificationStatus.VERIFIED


def test_fluent_hallucination_is_caught():
    """關鍵反例：語氣像、主題對、但來源完全沒提。這是 LLM 最常見的編造形式。"""
    text = "此段引用《黃帝內經》素問篇的五運六氣之說，闡明五行生剋的醫理基礎。"
    v = p.check_block(block(text), SLIDE, cfg(), "s#1", 0)
    assert v.status is VerificationStatus.UNVERIFIED
    assert "相似度" in v.reason or "具名實體" in v.reason


def test_fabricated_book_title_is_caught():
    """具名實體檢查：書名沒在來源出現就是編的。"""
    text = "一月為胞，精血凝也，此說見於《雲笈七籤》卷十一。"
    v = p.check_block(block(text), SLIDE, cfg(), "s#1", 0)
    assert v.status is VerificationStatus.UNVERIFIED
    assert "雲笈七籤" in str(v.missing_entities)


def test_fabricated_number_is_caught():
    """數字最容易被編，也最容易被讀者當真。"""
    text = "一月為胞，精血凝也，歷時約二十八日而成形。"
    v = p.check_block(block(text), SLIDE, cfg(), "s#1", 0)
    assert v.status is VerificationStatus.UNVERIFIED


def test_numbers_present_in_source_are_fine():
    text = "三月陽神為三魂，動而生也。"
    v = p.check_block(block(text, ContentType.SCRIPTURE), SLIDE, cfg(), "s#1", 0)
    assert not v.missing_entities


def test_transcript_provenance_checked_against_transcript():
    """口頭延伸的來源是逐字稿。此處刻意用「整合過」而非逐字照抄的表述——
    照抄會（正確地）被反向檢查判為 degenerate_copy，見下方測試。"""
    b = ContentBlock(
        type=ContentType.ORAL,
        text="講者以簽約作比喻，說明識蘊進入的時機。",
        provenance=Provenance(kind=ProvenanceKind.TRANSCRIPT, ref="0.0-90.0"),
    )
    assert p.check_block(b, TRANSCRIPT, cfg(), "s#1", 0).status is VerificationStatus.VERIFIED


def test_oral_block_copied_verbatim_from_transcript_is_degenerate():
    """口頭延伸應該是整合，不是把逐字稿搬過來。§5.5 #5。"""
    b = ContentBlock(
        type=ContentType.ORAL,
        text="講者用簽約來比喻識蘊進入的時機。",  # 與逐字稿一字不差
        provenance=Provenance(kind=ProvenanceKind.TRANSCRIPT, ref="0.0-90.0"),
    )
    v = p.check_block(b, TRANSCRIPT, cfg(), "s#1", 0)
    assert v.status is VerificationStatus.DEGENERATE_COPY


def test_content_taken_from_wrong_source_is_caught():
    """block 宣稱來自投影片，內容卻只出現在逐字稿裡——溯源就是要抓這個。"""
    v = p.check_block(block("講者用簽約來比喻識蘊進入的時機。"), SLIDE, cfg(), "s#1", 0)
    assert v.status is VerificationStatus.UNVERIFIED


# --------------------------------------------------------------------------
# 反向檢查：複製貼上同樣是失敗
# --------------------------------------------------------------------------


def test_lcs_ratio_detects_verbatim_copy():
    assert p.longest_common_substring_ratio("一月為胞，精血凝也。", SLIDE) == 1.0


def test_lcs_ratio_low_for_recombined_text():
    """重組過的文字即使用詞重疊高，最長連續共同片段仍短。
    這正是 LCS 比 n-gram 更適合判斷「是否只是搬運」的原因。"""
    text = "精血凝也是一月為胞的意思，形兆胚也則說二月為胎。"
    assert p.longest_common_substring_ratio(text, SLIDE) < 0.5


def test_scripture_may_be_verbatim():
    """§5.4：經文原文型別**應該**是高複製率（那是引文）。"""
    ir = _make_ir([("經文原文", "一月為胞，精血凝也。二月為胎，形兆胚也。")])
    verdict = p.check_video(ir, cfg())
    assert verdict.unverified == []


def test_vernacular_that_is_all_copy_is_degenerate():
    """§5.4 反向檢查：白話解說不該是逐字複製。
    §5.5 #5：不得為了通過溯源檢查而讓 LLM 只做複製貼上。"""
    ir = _make_ir(
        [
            ("白話解說", "一月為胞，精血凝也。二月為胎，形兆胚也。"),
            ("白話解說", "三月陽神為三魂，動而生也。四月陰靈為七魄，靜鎮形也。"),
        ]
    )
    verdict = p.check_video(ir, cfg())
    assert all(v.status is VerificationStatus.DEGENERATE_COPY for v in verdict.verdicts)


def test_occasional_quote_within_vernacular_is_tolerated():
    """白話解說中偶爾整句引用是正常的，只有**比例**超標才算退化。"""
    ir = _make_ir(
        [
            ("白話解說", "一月為胞，精血凝也。"),  # 逐字
            ("白話解說", "精血凝也是一月為胞的意思，形兆胚也則說二月為胎。"),
            ("白話解說", "三月時陽神化為三魂而主動，四月時陰靈化為七魄而主靜鎮形。"),
        ]
    )
    verdict = p.check_video(ir, cfg())
    assert verdict.unverified == []


# --------------------------------------------------------------------------
# 影片層級判定
# --------------------------------------------------------------------------


def _make_ir(blocks: list[tuple[str, str]]):
    from pathlib import Path
    from tempfile import mkdtemp

    from tests.factories import make_ir

    ir = make_ir(Path(mkdtemp()))
    ir.slides[0].slide_text = SLIDE
    seg = ir.segments[1]
    seg.transcript_corrected = TRANSCRIPT
    seg.understanding.content_blocks = [
        ContentBlock(
            type=ContentType(t),
            text=text,
            provenance=Provenance(kind=ProvenanceKind.SLIDE_OCR, ref="slide_001"),
        )
        for t, text in blocks
    ]
    return ir


def test_needs_review_when_unverified_ratio_exceeds_five_percent():
    """§5.4：unverified 比例 > 5% → 整支標記 needs_review，不進 chunks.jsonl。"""
    good = ("白話解說", "一月為胞是說精血凝聚成形，二月為胎則形兆已成胚。")
    bad = ("白話解說", "此處引用《黃帝內經》五運六氣之說，闡明五行生剋醫理。")
    ir = _make_ir([good] * 10 + [bad])
    verdict = p.check_video(ir, cfg())

    assert verdict.unverified_ratio > MAX_UNVERIFIED_RATIO
    assert verdict.needs_review is True
    assert ir.needs_review is True


def test_clean_video_does_not_need_review():
    good = ("白話解說", "一月為胞是說精血凝聚成形，二月為胎則形兆已成胚。")
    ir = _make_ir([good] * 10)
    verdict = p.check_video(ir, cfg())

    assert verdict.needs_review is False
    assert verdict.pass_rate == 1.0


def test_check_video_writes_verdict_back_into_blocks():
    """溯源結果要留在 IR 裡，debug markdown 才能標出哪些 block 可疑。"""
    ir = _make_ir([("白話解說", "此段引用《雲笈七籤》所載之說，與本經互為表裡。")])
    p.check_video(ir, cfg())
    block0 = ir.segments[1].understanding.content_blocks[0]
    assert block0.verification is VerificationStatus.UNVERIFIED
    assert block0.similarity is not None


def test_pass_rate_meets_threshold_definition():
    from weft.validation.thresholds import PROVENANCE_PASS_RATE

    good = ("白話解說", "一月為胞是說精血凝聚成形，二月為胎則形兆已成胚。")
    ir = _make_ir([good] * 20)
    assert p.check_video(ir, cfg()).pass_rate >= PROVENANCE_PASS_RATE


# --------------------------------------------------------------------------
# 具名實體抽取的邊界
# --------------------------------------------------------------------------


def test_person_names_are_not_auto_extracted():
    """刻意的限制：中文人名無邊界標記，regex 啟發式會把「太上老君」之類的
    詞誤判成人名，製造大量假 unverified，真幻覺反而被雜訊淹沒。
    人名溯源改由 S2b 的系列術語詞庫涵蓋（Phase 2）。
    """
    ents = p.extract_named_entities("太上老君說，一月為胞。")
    assert "太上老君" not in ents.all()


def test_book_titles_are_extracted():
    assert "雲笈七籤" in p.extract_named_entities("見於《雲笈七籤》卷十一").book_titles


def test_cjk_numbers_with_units_are_extracted():
    ents = p.extract_named_entities("歷時二十八日，共三卷。")
    assert ents.numbers


def test_unsupported_entities_lists_only_missing_ones():
    text = "一月為胞，精血凝也，見於《雲笈七籤》。"
    missing = p.unsupported_entities(text, SLIDE)
    assert "雲笈七籤" in missing


# --------------------------------------------------------------------------
# R12 校準的結論（docs/decisions.md D17、D18）
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("分為四種", "四"),          # 種
        ("佔七成以上", "七"),        # 成
        ("第五週開始", "五"),        # 週
        ("各需二十一天", "二十一"),  # 天
        ("體長三公分", "三"),        # 公分（多字單位）
        ("持續兩個月", "兩"),        # 個（原本就有）
    ],
)
def test_named_entity_units_cover_common_measures(text, expected):
    """單位清單不完整 = 永遠抓不到那些情形，而且**不會有任何測試變紅**。

    原本的清單是「年月日歲個張條章卷篇位次倍分」，漏掉週／天／種／成／公分。
    R12 校準時，口頭延伸的幻覺攔截率因此是 0/5；補上後升到 3/5。
    """
    assert expected in p.extract_named_entities(text).numbers


def test_multi_character_units_are_matched_before_single_ones():
    """`公分` 必須排在 `分` 之前。

    regex 的交替是**最左優先**不是最長優先——若 `分` 排在前面，「三公分」
    會被 `分` 匹配成功，位置對但語意錯（抽到的是「三公」後面的分）。
    """
    assert "三" in p.extract_named_entities("體長三公分").numbers


def test_forward_threshold_is_per_content_type():
    """§5.4 對反向檢查已明訂「依 type 分別設定範圍」，正向檢查原本卻只有
    單一門檻——這是規格的不對稱，D17 補上。

    門檻的高低反映的是**該型別與來源的關係有多緊密**，不是重要性：
    引文逐字重疊、改寫部分重疊、口語摘要落差大、版面描述可能跨語言。
    """
    by_type = ProvenanceConfig().min_similarity_by_type
    assert by_type["經文原文"] > by_type["白話解說"] > by_type["口頭延伸"]
    assert by_type["圖表描述"] == 0.0, (
        "圖表描述無法設下限——跨語言時忠實者本身就是 0.000"
    )


def test_scripture_still_has_a_meaningful_threshold():
    """經文原文是唯一 containment 有鑑別力的型別（實測分離 7.50x），
    它的門檻不該跟著其他型別一起放寬。"""
    assert ProvenanceConfig().min_similarity_by_type["經文原文"] >= 0.5


def test_expanded_paraphrase_of_terse_source_is_not_rejected():
    """來源比 block 短時，containment 有數學上限——文言文簡短句被展開成
    白話時必然如此。

    實測案例：來源「意導氣 氣成形 / 先天之氣:腎氣」12 字 ≈ 11 bigram，
    block 33 字 ≈ 32 bigram，理論上限只有 11/32 ≈ 0.34。
    """
    source = "意導氣 氣成形\n先天之氣:腎氣"
    block = ContentBlock(
        type=ContentType.VERNACULAR,
        text="意念引導氣的運行，氣進而形成形體。在先天狀態下，這股氣被稱為腎氣。",
        provenance=Provenance(kind=ProvenanceKind.SLIDE_OCR, ref="slide_001"),
    )
    verdict = p.check_block(block, source, cfg(), "s#1", 0)
    assert verdict.similarity < 0.25, "此案例的 containment 本來就低，前提變了請重新檢視"
    assert verdict.status is VerificationStatus.VERIFIED, (
        "逐型別門檻應讓這種「展開簡短來源」的忠實改寫通過"
    )


def test_cross_language_description_is_not_rejected():
    """跨語言時 containment 結構上必為 0——忠實與幻覺都是。

    實測案例：投影片是英文（Cleavage / Blastocyst / Implantation），
    block 是中文翻譯（卵裂／胚泡形成／著床）。這是 R12 校準中
    圖表描述分離度為 0.00x 的主因之一。
    """
    source = "Fertilized egg\nCleavage\nBlastocyst\nImplantation\nGastrulation"
    block = ContentBlock(
        type=ContentType.FIGURE,
        text="圖表展示受精卵發育的循環過程，包含卵裂、胚泡形成、著床與原腸胚形成等階段。",
        provenance=Provenance(kind=ProvenanceKind.SLIDE_OCR, ref="slide_001"),
    )
    verdict = p.check_block(block, source, cfg(), "s#1", 0)
    assert verdict.similarity == 0.0, "跨語言的 containment 應為 0；前提變了請重新檢視"
    assert verdict.status is VerificationStatus.VERIFIED, (
        "圖表描述的門檻應低到不誤殺跨語言的忠實翻譯"
    )


# --------------------------------------------------------------------------
# 票 01：溯源基準必須是 transcript_raw，不是校正後的版本
# --------------------------------------------------------------------------

#: 兩份逐字稿只差一個術語：ASR 把「識蘊」聽成「時運」。
#: 這是 R18 實測的典型形態——**等長同音替換**，不是隨機噪音。
_RAW = "這一段講的是時運進入的時機，父精母血在此時凝結成形，講者用簽約來比喻。"
_CORRECTED = _RAW.replace("時運", "識蘊")

#: 這幾條用 `經文原文`（門檻 0.60）而不是 `白話解說`（0.10）。
#: **不是為了讓測試好過**——R12 實測 containment 只對經文原文有鑑別力
#: （分離 7.50x），白話解說是 1.14x，門檻低到單一術語的差異推不動它。
#: 用一個本來就擋不住東西的型別去測「基準換了會怎樣」，測不到東西。
_QUOTE_TYPE = ContentType.SCRIPTURE


def _ir_with_correction(raw: str, corrected: str, block_text: str):
    """一支只有一段的 IR，那一段的逐字稿有 raw／corrected 兩個版本。

    刻意不用 `factories.make_ir`——這裡要控制的正是 raw 與 corrected 的差異，
    而工廠給的是一份「合法」資料。反例要精準地只動一個性質。
    """
    from weft.ir import (
        BoundaryMethod,
        Segment,
        SegmentMode,
        Understanding,
        VideoIR,
        VideoMeta,
    )

    seg = Segment(
        segment_id="v#000", video_id="v", t_start=0.0, t_end=30.0,
        mode=SegmentMode.TRANSCRIPT_ONLY, boundary_method=BoundaryMethod.VIDEO_BOUNDS,
        transcript_raw=raw, transcript_corrected=corrected,
        understanding=Understanding(
            is_slide=False, summary="",
            # **來源型別必須是 transcript**——這幾條測的正是逐字稿基準。
            # 用預設的 `block()`（slide_ocr）會走到另一條路：沒有投影片
            # → 來源為空 → 因「來源長度 0%」未通過，測不到要測的東西。
            content_blocks=[ContentBlock(
                type=_QUOTE_TYPE, text=block_text,
                provenance=Provenance(kind=ProvenanceKind.TRANSCRIPT, ref="0.0s-30.0s"),
            )],
        ),
    )
    return VideoIR(
        meta=VideoMeta(video_id="v", title="t", duration=30.0, url="u"),
        slides=[], segments=[seg],
    )


def test_baseline_is_raw_not_corrected():
    """校正後才對得上的內容**不得**被判為通過。

    這是 §5.4 獨立性的核心：S4b 用投影片校正逐字稿，再拿逐字稿驗證
    同樣來自投影片的內容，等於用來源 A 驗證來源 A。
    """
    ir = _ir_with_correction(_RAW, _CORRECTED, "識蘊進入")
    (v,) = p.check_video(ir, cfg()).verdicts
    assert v.status is VerificationStatus.UNVERIFIED, (
        "以校正後的逐字稿為基準時這一筆會通過——那正是要修掉的循環"
    )


def test_correction_dependent_block_is_flagged_not_silent():
    """未通過的同時要說得出「它是靠校正才對得上的」。

    票 01：「一段內容如果只有靠校正才對得上，會被標記出來，而不是靜默通過。」
    沒有這個標記，它會混在「兩個來源都對不上」裡，而那是完全不同的修法。
    """
    ir = _ir_with_correction(_RAW, _CORRECTED, "識蘊進入")
    verdict = p.check_video(ir, cfg())
    (v,) = verdict.verdicts
    assert v.depends_on_correction is True
    assert "校正" in v.reason
    assert verdict.depends_on_correction == [v]


def test_content_matching_raw_is_untouched():
    """對原始逐字稿本來就對得上的內容，不受這次改動影響。

    沒有這一條，上面兩個測試可以靠「把所有東西都判成未通過」通過。
    """
    ir = _ir_with_correction(_RAW, _CORRECTED, "父精母血在此時凝結")
    (v,) = p.check_video(ir, cfg()).verdicts
    assert v.status is VerificationStatus.VERIFIED
    assert v.depends_on_correction is False


def test_no_correction_means_no_flag():
    """raw 與 corrected 相同時不得標記——那一段根本沒有校正可依賴。"""
    ir = _ir_with_correction(_RAW, _RAW, "識蘊進入")
    (v,) = p.check_video(ir, cfg()).verdicts
    assert v.status is VerificationStatus.UNVERIFIED
    assert v.depends_on_correction is False


def test_unverified_jsonl_replaces_same_video_on_rerun(tmp_path):
    """重跑同一支影片不得讓 unverified.jsonl 累積重複紀錄。

    §5.6 的人工複核要從這個檔案數「有幾筆要看」。純附加模式下跑兩次
    就變成兩倍，而且新舊程式碼的紀錄混在一起分不出來。
    """
    import json

    from weft.stages.render import write_unverified

    path = tmp_path / "unverified.jsonl"
    ir = _ir_with_correction(_RAW, _CORRECTED, "識蘊進入")
    verdict = p.check_video(ir, cfg())
    assert verdict.unverified, "這份 fixture 要有未通過的條目，否則測不到東西"

    write_unverified(verdict, path)
    first = path.read_text(encoding="utf-8").splitlines()
    write_unverified(verdict, path)
    second = path.read_text(encoding="utf-8").splitlines()
    assert first == second, "重跑同一支影片後檔案內容應完全相同"

    # 別支影片的紀錄不得被清掉
    path.write_text("\n".join([*second,
        json.dumps({"video_id": "other", "segment_id": "other#000"})]) + "\n",
        encoding="utf-8")
    write_unverified(verdict, path)
    kept = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert any(r["video_id"] == "other" for r in kept)
    assert sum(1 for r in kept if r["video_id"] == "v") == len(verdict.unverified)
