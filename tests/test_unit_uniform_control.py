"""等距對照組與容忍窗覆蓋率（R37，2026-08-09）。

boundary F1 的分數有多少來自「切得準」、多少來自「切得多」，取決於
`±tolerance` 涵蓋了多少比例的時間軸。實測在分段上覆蓋率到 95% 時，
**不看內容的等距切拿到的 F1 比真正的方法還高**（0.459 vs 0.432）——
只看 F1 會得到「語意分段在 STEM 上沒用」，而 ±10s 下它的優勢是 +0.30。

這個檔案測的是**度量本身**，不是任何一個 pipeline 階段。
"""

from __future__ import annotations

import pytest

from weft.validation.metrics import (
    boundary_prf,
    margin_over_uniform,
    tolerance_coverage,
    uniform_boundaries,
)
from weft.validation.thresholds import MAX_TOLERANCE_COVERAGE


class TestUniformBoundaries:
    def test_cuts_are_evenly_spaced_and_inside_the_range(self):
        cuts = uniform_boundaries(0.0, 100.0, 3)
        assert cuts == pytest.approx([25.0, 50.0, 75.0])

    def test_no_cut_lands_on_an_endpoint(self):
        """端點是 body_start／body_end，不是邊界——落在那裡會平白多一個 TP。"""
        cuts = uniform_boundaries(10.0, 20.0, 5)
        assert all(10.0 < t < 20.0 for t in cuts)

    @pytest.mark.parametrize(("lo", "hi", "n"), [(0.0, 100.0, 0), (0.0, 100.0, -1),
                                                 (50.0, 50.0, 3), (60.0, 10.0, 3)])
    def test_degenerate_inputs_give_no_cuts(self, lo, hi, n):
        assert uniform_boundaries(lo, hi, n) == []


class TestToleranceCoverage:
    def test_matches_the_measured_numbers(self):
        """R37 表格裡的三支影片。改了公式這裡就會紅。"""
        assert tolerance_coverage(0.0, 14 * 60, 18, 20.0) == pytest.approx(0.857, abs=0.03)
        assert tolerance_coverage(0.0, 25 * 60, 30, 20.0) == pytest.approx(0.80, abs=0.03)
        assert tolerance_coverage(0.0, 40 * 60, 57, 20.0) == pytest.approx(0.95, abs=0.03)

    def test_saturates_at_one_rather_than_exceeding_it(self):
        """覆蓋率是比例。窗重疊時要飽和在 1，不能長成 3.7 那種讀不懂的數字。"""
        assert tolerance_coverage(0.0, 100.0, 50, 30.0) == 1.0

    def test_thirty_second_tolerance_is_degenerate_on_real_density(self):
        """實測：±30s 在三支影片上覆蓋率都是 100%。**那不是保守，是失效。**"""
        for body_min, cuts in ((14, 18), (25, 30), (40, 57)):
            assert tolerance_coverage(0.0, body_min * 60, cuts, 30.0) == 1.0

    def test_slide_detection_density_is_not_saturated(self):
        """換頁偵測在生產容忍窗（±2s）下覆蓋 5%，換成 ±20s 也才 46%。

        **飽和是容忍窗 × 刀數密度的性質**，不是容忍窗自己的：出事的分段
        刀數是換頁的 3–6 倍。這條在的理由是擋住「那就把
        `BOUNDARY_TOLERANCE_SEC` 全域改小」——那會把好的那一邊一起弄壞。
        """
        assert tolerance_coverage(0.0, 14 * 60, 10, 2.0) < 0.10
        assert tolerance_coverage(0.0, 14 * 60, 10, 20.0) < MAX_TOLERANCE_COVERAGE


class TestMarginOverUniform:
    def test_perfect_prediction_beats_the_control(self):
        truth = [40.0, 130.0, 155.0, 320.0]  # **刻意不等距**，見下一條
        f1, uniform_f1, _ = margin_over_uniform(truth, truth, 0.0, 400.0, 5.0)
        assert f1 == 1.0
        assert uniform_f1 < f1

    def test_evenly_spaced_truth_makes_the_control_unbeatable(self):
        """對照組的限制，記在這裡免得下次被當成 bug。

        真實邊界本身就等距時，「不看內容等距切」拿滿分——優勢是 0，
        而方法並沒有錯。第一版這個檔案就踩到了（`truth` 寫成 100/200/300
        而範圍是 0–400，那正好就是等距切的位置）。

        真實素材上不成立（實測講經 0.60–0.69 個/分、STEM 0.42 個/分，
        間隔都不規則），但**合成 fixture 很容易不小心構造出這種**。
        """
        truth = [100.0, 200.0, 300.0]
        f1, uniform_f1, _ = margin_over_uniform(truth, truth, 0.0, 400.0, 5.0)
        assert f1 == uniform_f1 == 1.0

    def test_the_control_uses_the_same_number_of_cuts(self):
        """對照組刀數不同 = 對照組在做不同難度的題目。

        票 12 就是這樣量出 +0.055 的假差距，而正確答案是 +0.188。
        """
        pred = [10.0, 20.0, 30.0, 40.0, 50.0]
        _, uniform_f1, _ = margin_over_uniform(pred, [25.0], 0.0, 100.0, 3.0)
        # 5 刀對 1 個真實邊界 → 對照組的 precision 上限是 1/5
        assert uniform_f1 <= 2 * (0.2 * 1.0) / (0.2 + 1.0) + 1e-9

    def test_a_saturated_window_hides_a_useless_method(self):
        """R37 的核心現象：窗夠寬時，**亂切也拿高分**。

        沒有這一條，「覆蓋率」就只是一個被算出來但沒人相信的數字。
        """
        truth = [100.0, 200.0, 300.0, 400.0, 500.0]
        garbage = [55.0, 155.0, 255.0, 355.0, 455.0]  # 每一刀都差 45 秒
        tight = boundary_prf(garbage, truth, 10.0).f1
        loose = boundary_prf(garbage, truth, 50.0).f1
        assert tight == 0.0, "±10s 下這組亂切應該一個都對不上"
        assert loose == 1.0, "±50s 下同一組亂切變成滿分"
        assert tolerance_coverage(0.0, 600.0, 5, 50.0) > MAX_TOLERANCE_COVERAGE

    def test_margin_can_be_negative(self):
        """方法輸給「不看內容」時要看得出來，不能夾到 0。

        實測 STEM 保留集 ±20s：方法 0.432、等距 0.459。
        """
        truth = [100.0, 200.0, 300.0]
        bad = [11.0, 12.0, 13.0]
        f1, uniform_f1, _ = margin_over_uniform(bad, truth, 0.0, 400.0, 5.0)
        assert f1 - uniform_f1 < 0
