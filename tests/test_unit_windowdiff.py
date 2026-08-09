"""WindowDiff（Pevzner & Hearst 2002）。2026-08-09。

**換掉 boundary F1 的理由必須被證明，不能只是換一個名字。**
R37 實測：±20s 的容忍窗在現行刀數下覆蓋 95% 的時間軸，於是
「切 3.4 倍的刀」與「切對」拿到的 F1 幾乎一樣（0.432 vs 0.459，
方法還輸給不看內容的等距切）。這個檔案裡的 `TestItCatchesWhatF1Misses`
就是拿同一組資料證明 WindowDiff 分得出來。

實作照 nltk `metrics.segmentation.windowdiff` 的定義，
但**沒有引入 nltk 相依**——它公布的三個期望值直接釘在這裡。
"""

from __future__ import annotations

import pytest

from weft.validation.metrics import (
    boundary_prf,
    boundary_string,
    default_window,
    window_diff,
)


class TestAgainstPublishedValues:
    """nltk docstring 裡的三個例子。**改壞了公式這裡就會紅。**"""

    @pytest.mark.parametrize(
        ("ref", "hyp", "expected"),
        [
            ("000100000010", "000100000010", 0.00),  # 完全相同
            ("000100000010", "000010000100", 0.30),  # 兩個邊界各偏一格
            ("000010000100", "100000010000", 0.80),  # 偏得更遠
        ],
    )
    def test_matches_nltk(self, ref, hyp, expected):
        assert window_diff(ref, hyp, 3) == pytest.approx(expected, abs=0.005)

    def test_identical_segmentations_score_zero(self):
        assert window_diff("0010010010", "0010010010", 2) == 0.0


class TestItCatchesWhatF1Misses:
    """**這個類別是換指標的全部理由。** 沒有它，WindowDiff 只是換個名字。"""

    def test_over_segmentation_is_penalised(self):
        """切 3 倍的刀要明顯比切對差。

        構造：參考每 10 格一刀，預測每 3 格一刀（約 3.3 倍）。
        """
        n = 60
        ref = "".join("1" if (i + 1) % 10 == 0 and i < n - 1 else "0" for i in range(n))
        over = "".join("1" if (i + 1) % 3 == 0 and i < n - 1 else "0" for i in range(n))
        k = default_window(ref)
        assert window_diff(ref, ref, k) == 0.0
        assert window_diff(ref, over, k) > 0.5, "切 3 倍的刀卻幾乎沒被罰"

    def test_boundary_f1_recall_is_completely_blind_to_over_segmentation(self):
        """對照組：同一份 3 倍過切，兩個指標各說什麼。

        實測 ±20s：`P=0.333 R=1.000 F1=0.500`，WindowDiff **0.721**。

        重點不是「F1 給滿分」（它給 0.500）——是**召回完全看不見過度分割**。
        多切的刀只要落在容忍窗裡就算命中，所以 R 恆為 1.000；
        只有 precision 在抵抗，而 R37 量到容忍窗一寬（覆蓋 95% 時間軸）
        連 precision 都被稀釋。

        WindowDiff 沒有這個結構：它比的是視窗內的**數量**，
        多切的每一刀都讓數量對不上。
        """
        truth = [100.0, 200.0, 300.0, 400.0, 500.0]
        over = [t + d for t in truth for d in (-15.0, 0.0, 15.0)]  # 每個真實邊界切三刀
        prf = boundary_prf(over, truth, 20.0)
        assert prf.recall == 1.0, "召回應該完全被騙過——這正是要展示的"
        assert prf.f1 == pytest.approx(0.500, abs=0.01)

        starts = [float(i) * 5 for i in range(120)]
        ref = boundary_string(truth, starts)
        hyp = boundary_string(over, starts)
        wd = window_diff(ref, hyp, default_window(ref))
        assert wd == pytest.approx(0.721, abs=0.01), "同一份資料，WindowDiff 判為很差"

    def test_under_segmentation_is_also_penalised(self):
        """反向也要罰，否則「一刀都不切」會變成最佳解。"""
        n = 60
        ref = "".join("1" if (i + 1) % 10 == 0 and i < n - 1 else "0" for i in range(n))
        nothing = "0" * n
        # k=5、每 10 格一刀 → 只有含參考邊界的視窗會被罰（5×5 / 56 ≈ 0.446）。
        # 釘住實際值而不是隨手訂一個門檻——後者會變成另一個沒有依據的數字。
        assert window_diff(ref, nothing, default_window(ref)) == pytest.approx(0.446, abs=0.01)

    def test_near_miss_costs_less_than_a_wild_miss(self):
        """偏一格要比偏很遠便宜——這是 Pk／WindowDiff 相對 F1 的另一半價值。"""
        ref = "0000010000010000"
        near = "0000100000100000"  # 各偏一格
        wild = "1000000000000010"  # 位置完全不對
        k = default_window(ref)
        assert window_diff(ref, near, k) < window_diff(ref, wild, k)

    def test_weighted_variant_punishes_over_segmentation_harder(self):
        """加權版不封頂在 1，過切越誇張差距越大。"""
        n = 60
        ref = "".join("1" if (i + 1) % 10 == 0 and i < n - 1 else "0" for i in range(n))
        over = "".join("1" if (i + 1) % 2 == 0 and i < n - 1 else "0" for i in range(n))
        k = default_window(ref)
        assert window_diff(ref, over, k, weighted=True) > window_diff(ref, over, k)


class TestDefaultWindow:
    """`k` 必須是**算出來的**，不是挑的。

    R37 的教訓：±20s 那個容忍窗沒有依據，而它讓整組量測失效。
    """

    def test_k_is_half_the_mean_reference_segment(self):
        # 60 格、5 個邊界 → 6 段、平均 10 格 → k = 5
        ref = "".join("1" if (i + 1) % 10 == 0 and i < 59 else "0" for i in range(60))
        assert ref.count("1") == 5
        assert default_window(ref) == 5

    def test_k_is_at_least_one_even_with_no_boundaries(self):
        assert default_window("0" * 20) >= 1


class TestBoundaryString:
    """秒 → 單位序列。轉換錯了，上面所有測試都在測空氣。"""

    def test_a_cut_marks_the_unit_it_falls_in(self):
        starts = [0.0, 10.0, 20.0, 30.0, 40.0]
        assert boundary_string([22.0], starts) == "00100"

    def test_multiple_cuts(self):
        starts = [0.0, 10.0, 20.0, 30.0, 40.0]
        assert boundary_string([12.0, 33.0], starts) == "01010"

    def test_a_cut_in_the_last_unit_is_dropped(self):
        """文末不是邊界。標了會平白多算一分，而參考那邊也不會標。"""
        starts = [0.0, 10.0, 20.0]
        assert boundary_string([25.0], starts) == "000"

    def test_no_cuts_gives_all_zeros(self):
        assert boundary_string([], [0.0, 10.0, 20.0]) == "000"

    def test_length_always_matches_the_units(self):
        starts = [float(i) * 5 for i in range(37)]
        assert len(boundary_string([3.0, 88.0, 150.0], starts)) == 37


class TestGuards:
    def test_unequal_lengths_raise(self):
        with pytest.raises(ValueError, match="長度不同"):
            window_diff("0010", "001", 2)

    def test_window_larger_than_segmentation_raises(self):
        with pytest.raises(ValueError, match="大於分段長度"):
            window_diff("0010", "0010", 99)
