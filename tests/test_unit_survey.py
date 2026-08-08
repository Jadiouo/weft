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


def test_speaker_ratio_deviation_is_recorded_not_aborted():
    """整片都是投影片（錄屏教學）——**偏離，但不中止**（票 04）。

    §4.0 自己寫「這個階段只回答這一支能不能用現有設計處理」，而
    「像不像中醫講經第 1 集」是另一件事。純螢幕錄影嚴重偏離講者佔比，
    但它是**更好處理**的情況——把它算成中止條件會讓整個系列跑不起來。
    """
    from weft.stages.survey import SPEAKER_RATIO_REFERENCE, deviation_notes

    screencast = profile(fullscreen_ratio=0.95)
    assert not any("講者佔比" in p for p in check_mismatches([screencast]))

    notes = deviation_notes([screencast], SPEAKER_RATIO_REFERENCE, "§1.3")
    assert any("講者佔比" in n for n in notes)
    assert any("不是中止條件" in n for n in notes)


def test_single_mode_material_does_not_abort_on_separation():
    """全片單一模式時「模式分離度」沒有意義，不得據此中止。

    純螢幕錄影（全程投影片）與純口播（全程講者）都會落在這裡，
    而兩者都比混合素材好處理。拿一個無意義的數字去中止是誤判。
    """
    assert check_mismatches([profile(fullscreen_ratio=0.99, mode_separation=1.0)]) == []
    assert check_mismatches([profile(fullscreen_ratio=0.01, mode_separation=1.0)]) == []
    # 混合素材仍然要擋
    assert any("分離度" in p
               for p in check_mismatches([profile(fullscreen_ratio=0.5, mode_separation=1.0)]))


def test_baseline_prefers_the_series_over_section_1_3(tmp_path):
    """基準是**本系列已跑過的影片**，只有第一支才退回 §1.3。

    §1.3 的 81.1% 是單支影片的觀察，v0.2 已把它降級為「範例，不是設計
    前提」。拿它當跨系列的通則，與 v0.1 把單支推廣成系列通則是同一個錯。
    """
    from weft.stages.survey import (
        SPEAKER_RATIO_REFERENCE,
        series_baseline,
        write_video_profile,
    )

    value, source = series_baseline(tmp_path, "PL_new")
    assert value == SPEAKER_RATIO_REFERENCE
    assert "§1.3" in source

    write_video_profile(profile(video_id="a", fullscreen_ratio=0.9, series_id="PL_new"),
                        tmp_path)
    write_video_profile(profile(video_id="b", fullscreen_ratio=0.8, series_id="PL_new"),
                        tmp_path)
    # 別的系列不得混進來
    write_video_profile(profile(video_id="c", fullscreen_ratio=0.1, series_id="PL_other"),
                        tmp_path)

    value, source = series_baseline(tmp_path, "PL_new")
    assert value == pytest.approx(0.15)  # (1-0.9 + 1-0.8) / 2
    assert "2 支" in source


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
    # 用混合素材（0.5）而非 0.95：單一模式會跳過分離度判準，
    # 而這一條要驗的是「多個問題會一次列全」。
    problems = check_mismatches([
        profile(fullscreen_ratio=0.5, mode_separation=1.0,
                sections_per_minute=10.0, transition_ratio=0.5)
    ])
    assert len(problems) == 3, problems


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


# ---------------------------------------------------------------------------
# v0.4：逐支跑 S-1，跨集比對攝影棚背景（SDD §4.0 第 5 條判準）
#
# 實測同一個播放清單的四集（水墨山景 ×2、木質牆 ×2）：
#   群內最大 0.029、跨群最小 0.096，分離 3.3x。門檻取 0.06。
# ---------------------------------------------------------------------------

def _profile(vid: str, fingerprint: list[float]):
    from weft.stages.survey import VideoProfile

    return VideoProfile(
        video_id=vid, duration=100.0, frame_count=100, fullscreen_ratio=0.2,
        mode_separation=5.0, otsu_cut=0.1, section_count=10,
        sections_per_minute=1.0, dwell_median=30.0, dwell_min=5.0, dwell_max=90.0,
        transition_frames=1, transition_ratio=0.01, camera_motion_frames=0,
        has_manual_caption=True, has_auto_caption=False,
        background_fingerprint=fingerprint,
    )


def test_same_backdrop_is_not_flagged():
    from weft.stages.survey import BACKGROUND_DRIFT, background_notes

    a = _profile("ep01", [0.5] * 64)
    b = _profile("ep05", [0.52] * 64)   # 差 0.02，群內
    assert background_notes([a, b]) == []


def test_new_backdrop_is_flagged_once():
    """換背景要報，但**同一個新背景不該每支都報**。

    實測第 14、27 集是同一面木質牆——第 14 集報一次就夠，
    第 27 集若再報一次，警告就變成雜訊。
    """
    from weft.stages.survey import background_notes

    ink1 = _profile("ep01", [0.5] * 64)
    ink2 = _profile("ep05", [0.52] * 64)
    wood1 = _profile("ep14", [0.7] * 64)   # 與水墨差 0.2
    wood2 = _profile("ep27", [0.72] * 64)  # 與 ep14 差 0.02

    notes = background_notes([ink1, ink2, wood1, wood2])
    assert len(notes) == 1
    assert "ep14" in notes[0]


def test_nearest_is_over_all_known_not_just_the_previous():
    """比對對象是**所有已知**，不是「前一支」。

    「前一支」在檔案系統上取決於檔名排序而非集數順序；
    而且系列可能在兩種背景之間來回切換。
    """
    from weft.stages.survey import BACKGROUND_DRIFT, nearest_background

    ink = _profile("ep01", [0.5] * 64)
    wood = _profile("ep14", [0.7] * 64)
    back_to_ink = _profile("ep20", [0.51] * 64)

    # 依序列位置，ep20 的「前一支」是 ep14（差 0.19），但它其實與 ep01 同背景
    video_id, distance = nearest_background(back_to_ink, [ink, wood])
    assert video_id == "ep01"
    assert distance < BACKGROUND_DRIFT


def test_missing_fingerprint_is_not_a_false_alarm():
    """舊的 profile 沒有指紋欄位——不該因此判定「換了背景」。"""
    from weft.stages.survey import background_notes, nearest_background

    old = _profile("ep01", [])
    new = _profile("ep02", [0.5] * 64)
    # 沒有指紋可比 → 不產生任何比對結果，更不能因此判定「換了背景」
    assert nearest_background(new, [old]) == None  # noqa: E711
    assert background_notes([old, new]) == []
