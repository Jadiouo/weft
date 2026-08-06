"""R13：術語校正的授權檢查。

背景：首跑實測 9 筆校正中有 4 筆超出授權（precision 0.56，§5.2 要求 ≥0.90）。
量測與門檻依據見 `experiments/r13_corrections/REPORT.md`。

**這裡的案例大多取自模型的實際輸出**，不是想像出來的失效模式。
"""

from __future__ import annotations

import pytest

from weft.ir import BoundaryMethod, Segment, SegmentMode
from weft.stages.understand import validate_corrections
from weft.validation.corrections import (
    MIN_PINYIN_SIMILARITY,
    pinyin_similarity,
    unauthorized_reason,
)


def _segment(transcript: str) -> Segment:
    return Segment(
        segment_id="vid#000",
        video_id="vid",
        t_start=0.0,
        t_end=10.0,
        mode=SegmentMode.SLIDE,
        candidate_ref="slide_000",
        cue_indices=[0],
        transcript_raw=transcript,
        transcript_corrected=transcript,
        boundary_method=BoundaryMethod.SLIDE_SWITCH,
    )


# ---------------------------------------------------------------------------
# 授權範圍內：這些**不能**被擋掉
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("from_text", "to_text"),
    [
        pytest.param("憍梵钵提", "憍梵波提", id="音近專名-模型實際輸出"),
        pytest.param("意地論", "瑜伽師地論", id="音近漏字-模型實際輸出"),
        pytest.param("學古學的", "學古文學的", id="音近漏字-模型實際輸出2"),
        pytest.param("家家當", "家當", id="疊字-模型實際輸出"),
        pytest.param("未", "未來", id="截斷補全-模型實際輸出"),
        pytest.param("時運", "識蘊", id="完全同音"),
        pytest.param("涅盤", "涅槃", id="同音錯字"),
        pytest.param("陰陽不劃", "陰陽布化", id="音近經文"),
        pytest.param("這這個", "這個", id="疊字"),
    ],
)
def test_authorized_corrections_survive(from_text, to_text):
    assert unauthorized_reason(from_text, to_text) is None


# ---------------------------------------------------------------------------
# 插入：`to` 補上了講者沒說的內容
#
# 這一類最危險——§5.3 不變量 10 只驗 `from` 在原文中，
# **驗不到 `to` 是否加了原文沒有的東西**。
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("from_text", "to_text"),
    [
        pytest.param("陽神為三魂", "陽神為三魂，動而生也", id="模型實際輸出"),
        pytest.param("一月為胞", "一月為胞，精血凝也", id="從投影片補全經文"),
        pytest.param("五蘊", "五蘊，色受想行識", id="附加解釋"),
        pytest.param("內觀經", "太上老君內觀經", id="補全書名"),
    ],
)
def test_insertion_is_rejected(from_text, to_text):
    reason = unauthorized_reason(from_text, to_text)
    assert reason is not None and reason.startswith("插入")


def test_insertion_rejected_even_though_from_is_in_transcript():
    """不變量 10 會放行這筆——`from` 確實在原文中。R13 這一道才擋得到。"""
    seg = _segment("所以三月陽神為三魂大家看這裡")
    raw = {"corrections": [
        {"from": "陽神為三魂", "to": "陽神為三魂，動而生也", "reason": "逐字稿漏掉經文註解"},
    ]}
    assert seg.transcript_raw.count("陽神為三魂") == 1  # 不變量 10 的條件成立
    assert validate_corrections(raw, seg) == []


# ---------------------------------------------------------------------------
# 語意改寫：讀音完全不同，不可能是聽寫錯誤
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("from_text", "to_text"),
    [
        pytest.param("投胎轉世", "十個月懷胎", id="模型實際輸出"),
        pytest.param("買房子", "受精卵著床", id="比喻換成本體"),
        pytest.param("他進來以後", "識蘊進入胎體之後", id="代詞展開成詮釋"),
    ],
)
def test_semantic_rewrite_is_rejected(from_text, to_text):
    reason = unauthorized_reason(from_text, to_text)
    assert reason is not None and reason.startswith("語意改寫")


def test_semantic_rewrite_rejected_end_to_end():
    seg = _segment("是你來找他們投胎轉世所以你長得怎麼樣")
    raw = {"corrections": [
        {"from": "投胎轉世", "to": "十個月懷胎", "reason": "根據上下文脈絡"},
    ]}
    assert validate_corrections(raw, seg) == []


# ---------------------------------------------------------------------------
# 已知擋不到的：事實修正
#
# 這不是「還沒做」，是**量測後判定程式端做不到**——
# `唐朝`→`宋朝` 拼音 0.500、`六祖惠能`→`六祖慧能` 拼音 1.000，
# 與授權樣本的分布完全重疊（授權最低 0.400）。
# 這個測試把缺口釘住：哪天有人宣稱補上了，這裡會紅燈提醒要重測。
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("from_text", "to_text"),
    [
        pytest.param("唐朝", "宋朝", id="改年代"),
        pytest.param("六祖惠能", "六祖慧能", id="編輯偏好"),
        pytest.param("三百年", "五百年", id="改數字"),
        pytest.param("好像很像", "很像", id="刪贅詞"),
    ],
)
def test_known_gap_factual_edits_are_not_caught(from_text, to_text):
    """記錄現況：這些違規只能靠 prompt 的負面示例與 §5.6 人工抽檢擋。"""
    assert unauthorized_reason(from_text, to_text) is None


def test_case_only_change_is_rejected():
    """`沒sense`→`沒Sense` 是模型的實際輸出。大小寫聽不出來，不是聽寫錯誤。"""
    reason = unauthorized_reason("沒sense", "沒Sense")
    assert reason is not None and reason.startswith("格式化")


def test_factual_correction_with_distinct_sound_is_caught_incidentally():
    """`啟示經`→`創世記` 讀音夠遠，會被語意改寫那道順便擋到。

    這是**巧合不是設計**——它被擋是因為 qi-shi-jing 與 chuang-shi-ji 差很多，
    不是因為系統認得出「這是事實修正」。
    """
    assert unauthorized_reason("啟示經", "創世記") is not None


# ---------------------------------------------------------------------------
# 拼音相似度本身
# ---------------------------------------------------------------------------

def test_pinyin_similarity_ignores_tone():
    """聲調不參與比對——講者口誤造成的近音錯字聲調常常不同。

    `座位` zuò-wèi 與 `作為` zuò-wéi 只差 `為` 的聲調。
    """
    assert pinyin_similarity("座位", "作為") == 1.0


def test_context_dependent_readings_lower_the_score():
    """已知限制：pypinyin 依詞判讀音，同一個字在不同詞裡讀法不同。

    `般若` 讀 bo-re（它認得佛典讀法），`波若` 讀 bo-ruo——`若` 的讀法不同，
    相似度只有 0.5。這類授權校正的分數會被壓低，但仍在門檻之上。
    餘裕變小是實情，記在這裡以免日後把門檻往上調時沒發現。
    """
    sim = pinyin_similarity("波若", "般若")
    assert sim == 0.5
    assert sim > MIN_PINYIN_SIMILARITY


def test_pinyin_threshold_has_margin_on_both_sides():
    """門檻不是貼齊實測邊界的。餘裕不足時這裡會紅燈。"""
    lowest_authorized = pinyin_similarity("意地論", "瑜伽師地論")
    highest_rewrite = pinyin_similarity("他進來以後", "識蘊進入胎體之後")
    assert highest_rewrite < MIN_PINYIN_SIMILARITY < lowest_authorized


def test_non_han_text_is_compared_as_is():
    assert pinyin_similarity("sensor", "sensor") == 1.0
    assert pinyin_similarity("sensor", "感測器") < MIN_PINYIN_SIMILARITY


# ---------------------------------------------------------------------------
# 既有的不變量 10 沒有被這一道取代
# ---------------------------------------------------------------------------

def test_invariant_10_still_applies():
    seg = _segment("完全無關的逐字稿")
    raw = {"corrections": [{"from": "時運", "to": "識蘊", "reason": "同音"}]}
    assert validate_corrections(raw, seg) == []


def test_authorized_correction_passes_end_to_end():
    seg = _segment("這個時運會先進入就是種子先搬過來")
    raw = {"corrections": [{"from": "時運", "to": "識蘊", "reason": "同音錯字"}]}
    out = validate_corrections(raw, seg)
    assert len(out) == 1
    assert (out[0].from_text, out[0].to_text) == ("時運", "識蘊")


# ---------------------------------------------------------------------------
# D22：套用校正必須冪等
#
# 首次實跑後重跑同一支影片，溯源通過率從 98.6% 掉到 47.9%——
# `未→未來` 被重複套用成 `未來來`、`未來來來`。
# ---------------------------------------------------------------------------

def test_applying_corrections_twice_gives_the_same_text():
    from weft.ir import Transcript, TranscriptCue, TranscriptSource
    from weft.stages.understand import apply_corrections

    cue = TranscriptCue(index=0, t_start=0.0, t_end=5.0, text_raw="現在、過去、未")
    transcript = Transcript(video_id="vid", source=TranscriptSource.MANUAL_CAPTION,
                            cues=[cue], raw_hash="x")
    seg = _segment("現在、過去、未")
    seg.cue_indices = [0]

    corrections = validate_corrections(
        {"corrections": [{"from": "未", "to": "未來", "reason": "後文已補全"}]}, seg)
    assert len(corrections) == 1

    apply_corrections(transcript, seg, corrections)
    once = cue.text_corrected
    assert once == "現在、過去、未來"

    apply_corrections(transcript, seg, corrections)
    assert cue.text_corrected == once, "重複套用改變了結果——§6.3 的續跑會壞掉"

    apply_corrections(transcript, seg, corrections)
    assert cue.text_corrected == once


def test_corrected_text_is_a_pure_function_of_raw():
    """`text_corrected` 帶著舊值進來時，必須被重新推導而不是疊加。"""
    from weft.ir import Transcript, TranscriptCue, TranscriptSource
    from weft.stages.understand import apply_corrections

    cue = TranscriptCue(index=0, t_start=0.0, t_end=5.0, text_raw="查憍梵钵提")
    cue.text_corrected = "查憍梵波提"  # 前一次執行留下的
    transcript = Transcript(video_id="vid", source=TranscriptSource.MANUAL_CAPTION,
                            cues=[cue], raw_hash="x")
    seg = _segment("查憍梵钵提")
    seg.cue_indices = [0]

    corrections = validate_corrections(
        {"corrections": [{"from": "憍梵钵提", "to": "憍梵波提", "reason": "異體字"}]}, seg)
    apply_corrections(transcript, seg, corrections)
    assert cue.text_corrected == "查憍梵波提"
    assert cue.text_raw == "查憍梵钵提", "§4.5 約束 3：text_raw 永不覆寫"


# ---------------------------------------------------------------------------
# D22 第二個 bug：`from` 是 `to` 的子字串時，全域取代會弄壞已經正確的文字
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("現在、過去、未來", "現在、過去、未來", id="已經正確-不可動"),
        pytest.param("一個就是未 這個叫未來", "一個就是未來 這個叫未來", id="一對一錯"),
        pytest.param("帶未來能用的", "帶未來能用的", id="全對"),
        pytest.param("未", "未來", id="只有錯的"),
    ],
)
def test_truncation_completion_does_not_corrupt_correct_text(raw, expected):
    from weft.stages.understand import _replace_outside_existing

    assert _replace_outside_existing(raw, "未", "未來") == expected


def test_replacement_is_idempotent_for_substring_corrections():
    from weft.stages.understand import _replace_outside_existing

    once = _replace_outside_existing("一個就是未 這個叫未來", "未", "未來")
    twice = _replace_outside_existing(once, "未", "未來")
    assert once == twice == "一個就是未來 這個叫未來"


def test_ordinary_correction_still_replaces_everywhere():
    """`from` 不是 `to` 的子字串時，行為不變——全部換掉。"""
    from weft.stages.understand import _replace_outside_existing

    assert _replace_outside_existing(
        "查憍梵钵提，憍梵钵提尊者", "憍梵钵提", "憍梵波提") == "查憍梵波提，憍梵波提尊者"
