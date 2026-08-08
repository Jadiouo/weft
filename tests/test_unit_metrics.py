"""SDD §5.2 指標計算的測試。

門檻靠指標，指標算錯的話門檻就是裝飾品。重點在**配對邏輯**——邊界偵測的
F1 完全取決於怎麼把 pred 配到 gt，這是最容易寫出「看起來對、數字虛高」的
地方。
"""

from __future__ import annotations

import pytest

from weft.validation import metrics as m
from weft.validation import thresholds as T


# --------------------------------------------------------------------------
# boundary F1
# --------------------------------------------------------------------------


def test_perfect_prediction_scores_one():
    gt = [30.0, 90.0, 135.0]
    assert m.boundary_prf(list(gt), gt).f1 == 1.0


def test_within_tolerance_counts_as_hit():
    gt = [30.0, 90.0]
    prf = m.boundary_prf([31.9, 88.1], gt)
    assert prf.tp == 2 and prf.f1 == 1.0


def test_outside_tolerance_is_miss():
    prf = m.boundary_prf([32.1], [30.0])
    assert prf.tp == 0 and prf.fp == 1 and prf.fn == 1


def test_extra_predictions_are_false_positives():
    """過度切分必須被罰——這正是 A5（內嵌影片切成數十頁）的失敗模式。"""
    prf = m.boundary_prf([30.0, 31.0, 32.0, 90.0], [30.0, 90.0])
    assert prf.tp == 2 and prf.fp == 2
    assert prf.precision == 0.5


def test_one_prediction_cannot_satisfy_two_ground_truths():
    """一對一配對：一個 pred 不能同時認領兩個相鄰 gt。"""
    prf = m.boundary_prf([30.0], [29.5, 30.5])
    assert prf.tp == 1 and prf.fn == 1


def test_greedy_pairing_prefers_globally_closest():
    """近距離優先配對。若用「對每個 gt 取最近 pred」的天真作法，
    gt=30.0 會先搶走 30.2，讓 29.9 配不到，虛報一個 FN。"""
    prf = m.boundary_prf([29.9, 30.2], [30.0, 31.0])
    assert prf.tp == 2


def test_empty_prediction_against_empty_truth_is_perfect():
    """A3 純講者：0 頁對 0 頁應為滿分，不是 0 分。"""
    assert m.boundary_prf([], []).f1 == 1.0


def test_false_positives_on_empty_truth_score_zero():
    """A3 若誤偵測出投影片，必須是 0 分。"""
    prf = m.boundary_prf([10.0, 20.0], [])
    assert prf.f1 == 0.0 and prf.fp == 2


def test_missing_all_boundaries_scores_zero():
    prf = m.boundary_prf([], [30.0, 60.0])
    assert prf.f1 == 0.0 and prf.fn == 2


def test_matched_deltas_are_signed():
    """位移方向要保留——系統性偏早或偏晚是可修的 bug，取絕對值就看不出來。"""
    prf = m.boundary_prf([31.0, 91.0], [30.0, 90.0])
    assert prf.matched_deltas == (1.0, 1.0)


def test_tolerance_default_matches_sdd():
    assert T.BOUNDARY_TOLERANCE_SEC == 2.0


# --------------------------------------------------------------------------
# 分類 accuracy
# --------------------------------------------------------------------------


def test_classification_accuracy_basic():
    assert m.classification_accuracy(["slide"] * 4, ["slide"] * 4) == 1.0
    assert m.classification_accuracy(["slide", "speaker"], ["slide", "slide"]) == 0.5


def test_classification_length_mismatch_raises():
    """抽幀數對不上是比分類錯更嚴重的 bug，不該被平均掉。"""
    with pytest.raises(ValueError, match="長度不符"):
        m.classification_accuracy(["slide"], ["slide", "speaker"])


def test_classification_empty_truth_raises():
    with pytest.raises(ValueError):
        m.classification_accuracy([], [])


# --------------------------------------------------------------------------
# 對齊誤差
# --------------------------------------------------------------------------


def test_median_absolute_error():
    # 誤差為 1.0 / 2.0 / 7.0，中位數 2.0
    assert m.median_absolute_error([1.0, 2.0, 10.0], [0.0, 0.0, 3.0]) == 2.0


def test_median_is_robust_to_single_outlier():
    """用中位數而非平均，正是為了不讓單一離群值主導。"""
    assert m.median_absolute_error([0.0, 0.0, 0.0, 100.0], [0.0, 0.0, 0.0, 0.0]) == 0.0


# --------------------------------------------------------------------------
# 術語校正
# --------------------------------------------------------------------------


def test_correction_precision_penalises_wrong_edits():
    """§5.2：術語校正 precision ≥ 0.90，寧可漏改，不可亂改。"""
    applied = [(2, "時運", "識蘊"), (5, "經文", "精文")]  # 第二筆是亂改
    expected = [(2, "時運", "識蘊")]
    prf = m.correction_prf(applied, expected)
    assert prf.precision == 0.5
    assert prf.precision < T.TERM_CORRECTION_PRECISION


def test_correction_missing_edit_hurts_recall_not_precision():
    """漏改只扣 recall。recall 在首版「記錄但不設硬門檻」，這正是設計意圖。"""
    prf = m.correction_prf([(2, "時運", "識蘊")], [(2, "時運", "識蘊"), (7, "時雲", "識蘊")])
    assert prf.precision == 1.0
    assert prf.recall == 0.5


def test_correction_right_word_wrong_position_is_wrong():
    prf = m.correction_prf([(9, "時運", "識蘊")], [(2, "時運", "識蘊")])
    assert prf.tp == 0


def test_no_corrections_expected_and_none_applied_is_perfect():
    """詞庫為空時 S2c 會跳過（§4.5 失敗行為），不該因此得 0 分。"""
    assert m.correction_prf([], []).f1 == 1.0


# --------------------------------------------------------------------------
# 逐條動畫合併（A2）
# --------------------------------------------------------------------------


def test_merge_accuracy_requires_exact_count():
    """§5.2：A2 合併正確率門檻為 1.00，不容失敗，故不做部分給分。"""
    assert m.merge_accuracy([1, 1, 1], [1, 1, 1]) == 1.0
    assert m.merge_accuracy([1, 6, 1], [1, 1, 1]) == pytest.approx(2 / 3)


def test_merge_over_merging_also_fails():
    """把兩頁併成一頁同樣是錯，不能只罰切太多。"""
    assert m.merge_accuracy([1, 0], [1, 1]) == 0.5


# --------------------------------------------------------------------------
# 門檻本身
# --------------------------------------------------------------------------


def test_thresholds_match_sdd_section_5_2():
    """§5.5 #7：不得為了讓測試通過而調低門檻。

    這條測試把 SDD §5.2 表格的數值抄一份在此。有人改 thresholds.py 就會
    紅燈，而要讓它變綠就必須連這裡一起改——改動因此無法混進一般 commit。
    """
    assert T.BOUNDARY_F1_SYNTHETIC == 0.95
    assert T.BOUNDARY_F1_REAL == 0.75
    assert T.SLIDE_CLASSIFICATION_ACCURACY == 0.95
    assert T.PROGRESSIVE_MERGE_ACCURACY == 1.00
    assert T.TERM_CORRECTION_PRECISION == 0.90
    assert T.TERM_CORRECTION_RECALL is None  # 記錄但不設硬門檻
    assert T.ALIGNMENT_MEDIAN_ERROR_SEC == 5.0
    assert T.PROVENANCE_PASS_RATE == 0.95
    assert T.MAX_UNVERIFIED_RATIO == 0.05


def test_thresholds_are_not_configurable():
    """門檻不得經設定檔覆寫——那等於留了一個調門檻的後門。"""
    from weft.config import Config

    dumped = str(Config().model_dump())
    for name in ("BOUNDARY_F1_SYNTHETIC", "boundary_f1", "frame_class_accuracy"):
        assert name not in dumped
