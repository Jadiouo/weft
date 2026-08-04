"""S-1 素材勘查。SDD §4.0。

S-1 存在的理由是「機械檢查抓不到素材與假設不符」。所以這裡最重要的測試
不是「數字算得對不對」，而是**它真的會在不符時中止**——一個永遠回傳
「相符」的勘查階段比沒有還糟，因為它會讓人以為檢查過了。
"""

from __future__ import annotations

import pytest

from weft.stages.survey import (
    MAX_MIDBAND_RATIO,
    MAX_SECTIONS_PER_MINUTE,
    MIN_MODE_SEPARATION,
    SPEAKER_RATIO_REFERENCE,
    SPEAKER_RATIO_TOLERANCE,
    SeriesProfile,
    VideoProfile,
    check_mismatches,
)


def profile(**overrides) -> VideoProfile:
    """一份與 §1.3 相符的 profile（取自 zIglvjoU9vo 的實測值）。"""
    base = dict(
        video_id="v", duration=2520.0, frame_count=2519,
        fullscreen_ratio=0.185, mode_separation=7.62, otsu_cut=0.1665,
        section_count=49, sections_per_minute=1.17,
        dwell_median=13.0, dwell_min=2.0, dwell_max=313.0,
        transition_frames=21, transition_ratio=0.0083,
        camera_motion_frames=15,
        has_manual_caption=True, has_auto_caption=False,
    )
    base.update(overrides)
    return VideoProfile(**base)


# --------------------------------------------------------------------------
# 正例：實測的真實素材必須通過
# --------------------------------------------------------------------------


def test_real_material_profile_passes():
    """zIglvjoU9vo 的實測值必須通過——否則 S-1 會擋掉本專案的目標素材。"""
    assert check_mismatches([profile()]) == []


def test_empty_input_is_a_mismatch():
    """沒有影片可勘查不是「相符」。"""
    assert check_mismatches([]) != []


# --------------------------------------------------------------------------
# 反例：四條中止條件各自可被觸發
# --------------------------------------------------------------------------


def test_speaker_ratio_far_from_reference_is_flagged():
    """整片都是投影片（例如錄屏教學）——畫面結構與現有設計的假設不同。"""
    problems = check_mismatches([profile(fullscreen_ratio=0.95)])
    assert any("講者佔比" in p for p in problems)


def test_speaker_ratio_within_tolerance_passes():
    edge = 1.0 - (SPEAKER_RATIO_REFERENCE - SPEAKER_RATIO_TOLERANCE + 0.01)
    assert check_mismatches([profile(fullscreen_ratio=edge)]) == []


def test_low_mode_separation_is_flagged():
    """兩種畫面模式分不開 → §4.3 步驟 2 的二分類不可靠。"""
    problems = check_mismatches([profile(mode_separation=MIN_MODE_SEPARATION - 0.5)])
    assert any("分離度" in p for p in problems)


def test_separation_uses_the_worst_video_not_the_average():
    """一支不合格就該擋——平均會讓好片掩蓋壞片。"""
    problems = check_mismatches([profile(), profile(video_id="w", mode_separation=1.2)])
    assert any("分離度" in p for p in problems)


def test_over_segmentation_is_flagged_with_cost_estimate():
    """區段密度直接決定 VLM 請求數。訊息必須告訴使用者會花多少。"""
    problems = check_mismatches([
        profile(sections_per_minute=MAX_SECTIONS_PER_MINUTE + 2, section_count=300)
    ])
    assert any("區段密度" in p for p in problems)
    assert any("VLM 請求" in p for p in problems), "沒有估算成本，使用者無從判斷嚴重性"


def test_too_many_midband_frames_is_flagged():
    """中間帶太多 → 畫面模式可能不只兩種，§1.3 的二分假設不成立。"""
    problems = check_mismatches([profile(transition_ratio=MAX_MIDBAND_RATIO + 0.05)])
    assert any("兩種模式之間" in p for p in problems)


def test_multiple_problems_are_all_reported():
    """一次列全，不要修一個才發現下一個。"""
    problems = check_mismatches([
        profile(fullscreen_ratio=0.95, mode_separation=1.0,
                sections_per_minute=10.0, transition_ratio=0.5)
    ])
    assert len(problems) == 4


# --------------------------------------------------------------------------
# 中止語意
# --------------------------------------------------------------------------


def test_series_profile_ok_reflects_mismatches():
    assert SeriesProfile(series_id="PL", videos=[profile()], mismatches=[]).ok is True
    assert SeriesProfile(series_id="PL", videos=[profile()], mismatches=["x"]).ok is False


def test_survey_does_not_produce_knowledge_base_content():
    """§4.0：「這個階段不產生知識庫內容。」

    機械式護欄：survey 模組不得寫出 chunks、understanding 或呼叫雲端模型。
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src/weft/stages/survey.py").read_text(
        encoding="utf-8"
    )
    forbidden = {"google", "genai", "openai", "anthropic"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {(getattr(node, "module", "") or "").split(".")[0]}
            names |= {a.name.split(".")[0] for a in node.names}
            assert not (names & forbidden), f"survey.py 引入了雲端客戶端：{names & forbidden}"
    assert "chunks" not in source.lower()


def test_thresholds_are_material_fit_not_acceptance_thresholds():
    """§4.0 的中止條件量的是「素材適不適用」，不是 §5.2 的「演算法準不準」。
    它們因此**不在** ACCEPTANCE_THRESHOLDS 中，可以隨素材調整。"""
    from weft.validation.thresholds import ACCEPTANCE_THRESHOLDS

    for name in ("MIN_MODE_SEPARATION", "MAX_SECTIONS_PER_MINUTE",
                 "SPEAKER_RATIO_REFERENCE", "MAX_MIDBAND_RATIO"):
        assert name not in ACCEPTANCE_THRESHOLDS
