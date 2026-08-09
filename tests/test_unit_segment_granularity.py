"""分段粒度守衛（R40，2026-08-09）。

R40 量到一件足以否定整個 S3 的事：**Hearst 1997 的原始 α = −0.5 在三支
真實影片上都輸給「整支影片當一段」**（WindowDiff 0.529／0.490／0.772
vs 一刀不切的 0.451／0.464／0.467）。分段在做負功，而先前所有量測
（boundary F1）都看不出來——因為 F1 對過度分割結構上不敏感。

這個檔案守的是**那件事不會再悄悄回來**：

- α 的語意正確（越大越少刀），改壞方向會紅
- 生產值就是 R40 選出來的那個，被人改掉會紅
- **合成素材上，現行 α 產出的刀數不會誇張地超過真實邊界數**

前兩條是機械的。第三條才是真的驗收，而它刻意用合成素材——
真實影片的黃金集驗收在 `test_e2e_pipeline.py`，需要 `work/`。
"""

from __future__ import annotations

import numpy as np
import pytest

from weft.stages.segment import DEPTH_ALPHA, depth_cut_indices, topic_boundaries
from weft.validation.metrics import (
    boundary_string,
    default_window,
    window_diff,
)


def _cues(texts: list[str], seconds: float = 5.0):
    """把一串文字做成等長的 cue。用真的 `TranscriptCue`，不是 stub。"""
    from weft.ir import TranscriptCue

    return [
        TranscriptCue(index=i, t_start=i * seconds, t_end=(i + 1) * seconds,
                      text_raw=t)
        for i, t in enumerate(texts)
    ]


class TestAlphaSemantics:
    def test_larger_alpha_never_cuts_more(self):
        """α 越大門檻越嚴 → 刀數單調不增。**方向弄反了這條會紅。**"""
        rng = np.random.default_rng(0)
        scores = rng.random(200)
        counts = [len(depth_cut_indices(scores, a))
                  for a in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0)]
        assert counts == sorted(counts, reverse=True), f"刀數不是單調遞減：{counts}"

    def test_the_original_hearst_setting_cuts_more_than_ours(self):
        """R40 的整個前提：−0.5 比 +0.75 切得多。

        沒有這條，「我們調嚴了門檻」這句話沒有任何東西撐著。
        """
        rng = np.random.default_rng(1)
        scores = rng.random(300)
        assert len(depth_cut_indices(scores, -0.5)) > len(
            depth_cut_indices(scores, DEPTH_ALPHA)
        )

    def test_production_alpha_is_the_measured_one(self):
        """改這個常數會讓所有 segment_id 位移、S4c 快取全部失效（D32）。

        它不該混在一般 commit 裡。R40 選的是 +0.75。
        """
        assert DEPTH_ALPHA == 0.75


class TestItActuallyReducesOverSegmentation:
    """**這個類別是改 α 的全部理由。** 前面那些只證明旋鈕會轉。"""

    @staticmethod
    def _material() -> tuple[list, list[float]]:
        """六個話題，每個 12 句，用不重疊的字集——真實邊界在句 12/24/36/48/60。

        刻意讓段長相等：這裡測的是**刀數**，段長變異另外在真實素材上量
        （R40 §3，合成素材做不出 4.3–7.7 倍的分佈）。
        """
        topics = ["脈診寸關尺浮沉遲數", "針灸經絡穴位補瀉手法",
                  "方劑君臣佐使配伍禁忌", "舌診苔色質地厚薄潤燥",
                  "藏象心肝脾肺腎功能", "六淫風寒暑濕燥火致病"]
        texts = [t[i % len(t)] * 8 + t for t in topics for i in range(12)]
        cues = _cues(texts)
        truth = [cues[i].t_start for i in (12, 24, 36, 48, 60)]
        return cues, truth

    def test_current_alpha_does_not_wildly_over_segment(self):
        """現行設定的刀數不得超過真實邊界數的 2 倍。

        R40 在真實 STEM 素材上量到 α=−0.5 時是 **3.4 倍**，而那個設定
        輸給「一刀不切」。2.0 是合成素材上的守門值，**不是**真實素材的
        驗收門檻（那個在 R40，且 STEM 至今未達標）。
        """
        cues, truth = self._material()
        cuts = topic_boundaries(cues, block_chars=40, window=3)
        assert cuts, "一刀都沒切——這條測不到東西"
        assert len(cuts) <= 2 * len(truth), (
            f"切了 {len(cuts)} 刀，真實邊界 {len(truth)} 個"
        )

    def test_current_alpha_beats_the_original_on_windowdiff(self):
        """+0.75 要比 −0.5 好，否則這次改動沒有理由。

        用 WindowDiff 而不是 boundary F1——R40 實測同一批改動在 F1 上
        幾乎不動（0.429→0.421），在 WindowDiff 上是 0.529→0.360。
        """
        cues, truth = self._material()
        units = [c.t_start for c in cues]
        ref = boundary_string(truth, units)
        k = default_window(ref)

        old = boundary_string(
            topic_boundaries(cues, 40, 3, alpha=-0.5), units)
        new = boundary_string(
            topic_boundaries(cues, 40, 3, alpha=DEPTH_ALPHA), units)
        assert window_diff(ref, new, k) <= window_diff(ref, old, k)

    def test_beats_not_segmenting_at_all(self):
        """**這是 R40 揭穿舊設定的那條線。**

        「整支影片當一段」是免費的下界。分段贏不過它就是在做負功，
        而真實素材上舊設定三支全輸。
        """
        cues, truth = self._material()
        units = [c.t_start for c in cues]
        ref = boundary_string(truth, units)
        k = default_window(ref)
        nothing = "0" * len(units)
        ours = boundary_string(topic_boundaries(cues, 40, 3), units)
        assert window_diff(ref, ours, k) < window_diff(ref, nothing, k), (
            "分段的 WindowDiff 比完全不分段還差"
        )


class TestAlphaIsWiredThrough:
    def test_topic_boundaries_honours_the_argument(self):
        """參數沒接上的話，上面所有比較都在比同一組結果。"""
        cues, _ = TestItActuallyReducesOverSegmentation._material()
        assert (topic_boundaries(cues, 40, 3, alpha=-1.0)
                != topic_boundaries(cues, 40, 3, alpha=2.0))

    def test_default_matches_the_constant(self):
        cues, _ = TestItActuallyReducesOverSegmentation._material()
        assert (topic_boundaries(cues, 40, 3)
                == topic_boundaries(cues, 40, 3, alpha=DEPTH_ALPHA))


@pytest.mark.parametrize("alpha", [-0.5, 0.0, 0.75, 1.5])
def test_no_crash_on_degenerate_scores(alpha):
    """全部相同的分數 → std=0。除以零或全切都不行。"""
    assert depth_cut_indices(np.full(50, 0.5), alpha) == []
