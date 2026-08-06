"""S4b 詞庫事後校正（D25）。

R20 實測：拼音門檻 0.90 下 precision 90%，抓到 14 個相異錯誤；
對照組是 VLM 在人工字幕上產生的 7 筆。案例取自那次量測的實際結果。
"""

from __future__ import annotations

import pytest

from weft.config import S4bConfig
from weft.ir import (
    BoundaryMethod,
    CorrectionMethod,
    Segment,
    SegmentMode,
    Slide,
    Transcript,
    TranscriptCue,
    TranscriptSource,
)
from weft.stages.lexicon import apply_to_video, build_lexicon, corrections_for, propose


def _slide(text: str, sid: str = "slide_001") -> Slide:
    return Slide(slide_id=sid, image_path=f"03_slides/{sid}.png",
                 t_first_seen=0.0, t_last_seen=10.0, slide_text=text)


def _segment(raw: str) -> Segment:
    return Segment(
        segment_id="vid#000", video_id="vid", t_start=0.0, t_end=10.0,
        mode=SegmentMode.SLIDE, candidate_ref="slide_001", cue_indices=[0],
        transcript_raw=raw, transcript_corrected=raw,
        boundary_method=BoundaryMethod.SLIDE_SWITCH,
    )


# ---------------------------------------------------------------------------
# 詞庫建立
# ---------------------------------------------------------------------------

def test_lexicon_takes_delimited_units_not_ngrams():
    """詞庫是標點分隔的**完整**片段，不是任意 n-gram（R20 實測）。

    任意 n-gram 會把某行的碎片當術語，於是把正確的逐字稿改壞
    （`一開始`→`以開始`）。人工字幕上的誤報 19 → 5，Whisper 上的
    precision 91% → 100%。
    """
    lex = build_lexicon([_slide("五月五行分藏，以安神也。\nAdult / Blastocyst")])
    assert "五月五行分藏" in lex
    assert "以安神也" in lex
    # **碎片不得進詞庫**——這正是 n-gram 版的病灶
    assert "五行分藏" not in lex
    assert "以安神" not in lex
    # 標點不得跨越
    assert "分藏以安" not in lex
    # 英數不進詞庫
    assert not any(any(c.isascii() for c in t) for t in lex)


def test_lexicon_length_bounds():
    lex = build_lexicon([_slide("太上老君內觀經\n甲乙\n" + "長" * 20)])
    assert "太上老君內觀經" in lex
    assert "甲乙" not in lex          # 2 字太短，同音碎片太多
    assert "長" * 20 not in lex       # 太長，不會被整段聽錯


def test_lexicon_spans_all_slides():
    """詞庫必須跨全部投影片——第 5 張的術語可能在第 20 段被聽錯。"""
    lex = build_lexicon([_slide("七精開竅", "slide_001"),
                         _slide("宮室羅布", "slide_020")])
    assert "七精開竅" in lex and "宮室羅布" in lex


def test_fragments_do_not_corrupt_correct_text():
    """R20 抓到的實際回歸：`一開始` 不得被改成投影片碎片 `以開始`。"""
    lex = build_lexicon([_slide("醫學之光，以開始探索。")])
    assert propose("我們一開始就講了", lex, 0.90) == []


# ---------------------------------------------------------------------------
# 提案：R20 量測到的真實錯誤
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("wrong", "right"),
    [
        pytest.param("六律定府", "六律定腑", id="R20-實測"),
        pytest.param("七精開竅", "七精開竅", id="已正確-不提案"),
        pytest.param("天地夠精", "天地媾精", id="R20-實測2"),
        pytest.param("宮室羅布", "宮室羅布", id="已正確-不提案2"),
    ],
)
def test_proposes_homophone_variants(wrong, right):
    out = propose(f"所以{wrong}然後", {right}, 0.90)
    if wrong == right:
        assert out == [], "已經寫對的不該被提案"
    else:
        assert (wrong, right) in [(a, b) for a, b, _s in out]


def test_does_not_propose_when_term_already_present():
    """詞庫中的術語若已出現在文字裡，整個術語跳過——連別處的音近片段也不掃。"""
    assert propose("五行分藏又提到五行分障", {"五行分藏"}, 0.90) == []


def test_low_similarity_is_not_proposed():
    """R20：誤報集中在 0.70–0.80，0.90 以上實際上都是完全同音。"""
    assert propose("這個房子要裝潢", {"意導氣成形"}, 0.90) == []


# ---------------------------------------------------------------------------
# 仍走 R13 的三道閘
# ---------------------------------------------------------------------------

def test_r13_gates_still_apply():
    """詞庫提案不繞過任何既有把關。

    `陽神為三魂` → `陽神為三魂動而生也` 是 R13 擋掉的插入型；
    即使詞庫裡有後者，也不得放行。
    """
    seg = _segment("三月陽神為三魂大家看")
    cfg = S4bConfig()
    out = corrections_for(seg, {"陽神為三魂動而生也"}, cfg)
    assert all(c.to_text != "陽神為三魂動而生也" for c in out)


def test_corrections_are_tagged_as_lexicon():
    """來源必須可稽核——§5.6 要分得出哪些是 VLM 提的、哪些是詞庫掃的。"""
    seg = _segment("所以六律定府然後")
    out = corrections_for(seg, {"六律定腑"}, S4bConfig())
    assert len(out) == 1
    assert out[0].method is CorrectionMethod.LEXICON
    assert out[0].source == "lexicon"
    assert out[0].score is not None


def test_does_not_duplicate_existing_vlm_corrections():
    seg = _segment("所以六律定府然後")
    seg.corrections = [
        __import__("weft.ir", fromlist=["Correction"]).Correction(
            **{"from": "六律定府", "to": "六律定腑", "source": "slide_001"})
    ]
    assert corrections_for(seg, {"六律定腑"}, S4bConfig()) == []


# ---------------------------------------------------------------------------
# 整片套用
# ---------------------------------------------------------------------------

def test_apply_to_video_updates_transcript():
    cue = TranscriptCue(index=0, t_start=0.0, t_end=10.0, text_raw="六月六律定府用滋靈也")
    transcript = Transcript(video_id="vid", source=TranscriptSource.WHISPER,
                            cues=[cue], raw_hash="x")
    seg = _segment(cue.text_raw)
    seg.cue_indices = [0]
    added = apply_to_video([seg], [_slide("六月六律定腑，用滋靈也。")], transcript, S4bConfig())
    assert added >= 1
    assert "六律定腑" in cue.text_corrected
    assert cue.text_raw == "六月六律定府用滋靈也", "§4.5 約束 3：text_raw 永不覆寫"


def test_disabled_is_a_noop():
    cue = TranscriptCue(index=0, t_start=0.0, t_end=10.0, text_raw="六月六律定府")
    transcript = Transcript(video_id="vid", source=TranscriptSource.WHISPER,
                            cues=[cue], raw_hash="x")
    seg = _segment(cue.text_raw)
    cfg = S4bConfig(enabled=False)
    assert apply_to_video([seg], [_slide("六律定腑")], transcript, cfg) == 0


def test_empty_lexicon_is_a_noop():
    cue = TranscriptCue(index=0, t_start=0.0, t_end=10.0, text_raw="六月六律定府")
    transcript = Transcript(video_id="vid", source=TranscriptSource.WHISPER,
                            cues=[cue], raw_hash="x")
    seg = _segment(cue.text_raw)
    assert apply_to_video([seg], [_slide("")], transcript, S4bConfig()) == 0
