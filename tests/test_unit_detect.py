"""換頁偵測模組的單元測試。SDD §4.3 步驟 4–6。

e2e 測試（`test_e2e_pipeline.py`）驗證的是「在真影片上分數夠不夠」；這裡
驗證的是**模型的性質**——那些一旦破壞就會在真實素材上悄悄失效、但合成
素材可能剛好還過得去的東西。

§5.5 #1：不得以固定間隔取樣冒充換頁偵測。

**v0.3 移除了 speaker/slide 分類**——CV 只負責找靜止區段，分類交給 S4 的
VLM（見 docs/decisions.md D16）。原本針對 `split_runs` 的測試因此移除。
"""

from __future__ import annotations

import numpy as np
import pytest

from weft.stages.detect import (
    BULK_QUANTILE,
    KEYFRAME_EDGE_MARGIN,
    Section,
    ink_jaccard,
    merge_progressive,
    viterbi_changes,
)
from weft.stages.frames import Frame, ink_containment


def frame(index: int, ink: np.ndarray | None = None) -> Frame:
    blank = np.zeros((4, 4), dtype=bool) if ink is None else ink
    return Frame(
        index=index,
        t=index + 0.5,
        feature=np.zeros((4, 4), dtype=np.float32),
        ink=blank,
    )


def mask(rows: list[str]) -> np.ndarray:
    """用字元畫 ink 遮罩，`#` 為前景。讓測試意圖一眼可見。"""
    return np.array([[c == "#" for c in row] for row in rows], dtype=bool)


# --------------------------------------------------------------------------
# ink Jaccard
# --------------------------------------------------------------------------


def test_identical_masks_have_zero_distance():
    m = mask(["##..", "..##"])
    assert ink_jaccard(m, m) == 0.0


def test_disjoint_masks_have_max_distance():
    assert ink_jaccard(mask(["##.."]), mask(["..##"])) == 1.0


def test_two_blank_frames_are_not_a_change():
    """兩張空白頁之間沒有換頁。若回傳 1.0（union 為 0 時除以零的自然結果），
    整段空白畫面會被切成無數頁。"""
    blank = np.zeros((4, 4), dtype=bool)
    assert ink_jaccard(blank, blank) == 0.0


def test_partial_overlap_is_between():
    d = ink_jaccard(mask(["###."]), mask([".###"]))
    assert 0.0 < d < 1.0


# --------------------------------------------------------------------------
# 視覺包含（逐條動畫的判準）
# --------------------------------------------------------------------------


def test_containment_is_one_when_content_only_grows():
    """逐條動畫：舊內容位置不變，新內容疊上去。"""
    before = mask(["##..", "....", "...."])
    after = mask(["##..", "##..", "...."])
    assert ink_containment(before, after) == 1.0


def test_containment_is_low_for_a_different_page():
    before = mask(["##..", "##..", "...."])
    after = mask(["...."] * 2 + ["..##"])
    assert ink_containment(before, after) < 0.2


def test_containment_is_asymmetric():
    """方向性是重點：build 是「舊 ⊂ 新」，反過來不成立。"""
    before = mask(["##..", "...."])
    after = mask(["##..", "##.."])
    assert ink_containment(before, after) == 1.0
    assert ink_containment(after, before) == 0.5


def test_containment_drops_when_content_is_removed():
    """內容變少不是 build（可能是回放到前一頁）。containment 會自動反映，
    不需要額外的「內容必須增加」條件。"""
    before = mask(["##..", "##.."])
    after = mask(["##..", "...."])
    assert ink_containment(before, after) == 0.5


# --------------------------------------------------------------------------
# HMM
# --------------------------------------------------------------------------


def test_no_changes_detected_in_flat_sequence():
    """對抗樣本 A4：整支影片只有一頁 + 移動的雷射筆。
    純相對的離群偵測會把量化雜訊當成換頁——`min_ink_change` 就是為此存在。"""
    d = np.full(90, 0.002)
    d[45] = 0.03  # 雷射筆掃過造成的小擾動
    assert viterbi_changes(d, 0.97, 0.01) == []


def test_obvious_change_is_detected():
    d = np.full(60, 0.005)
    d[30] = 0.8
    assert viterbi_changes(d, 0.97, 0.01) == [30]


def test_changes_are_never_consecutive():
    """`CHANGE → CHANGE` 機率為 0 是模型的關鍵。沒有它，高頻抖動
    （對抗樣本 A5 的內嵌影片）會被解釋成「每一幀都在換頁」。"""
    d = np.full(40, 0.9)  # 每一幀看起來都像換頁
    changes = viterbi_changes(d, 0.97, 0.01)
    assert all(b - a > 1 for a, b in zip(changes, changes[1:]))


def test_sustained_moderate_noise_is_not_a_change():
    """A5 的內嵌影片：持續的中等幅度變化。發射模型的尺度由資料自身估計，
    所以基線會自動抬高，這些不該被判為換頁。"""
    d = np.full(100, 0.08)
    d[20] = 0.99
    d[80] = 0.97
    assert viterbi_changes(d, 0.97, 0.01) == [20, 80]


def test_emission_scale_adapts_to_the_video():
    """**同一個絕對距離**在不同影片中得到不同判定——這正是「避免手調門檻」
    的意思：尺度來自資料，不是常數。

    0.15 在雜訊基線 0.08 的片源裡只是抖動；在基線 0.001 的乾淨片源裡
    是明確的換頁。任何固定門檻都無法同時做對這兩件事。
    """
    noisy = np.full(100, 0.08)
    noisy[50] = 0.15
    assert viterbi_changes(noisy, 0.97, 0.01) == []

    clean = np.full(100, 0.001)
    clean[50] = 0.15
    assert viterbi_changes(clean, 0.97, 0.01) == [50]


def test_small_ink_change_is_never_a_page_turn():
    """即使在完全無雜訊的片源，只改變幾個百分點的 ink 也不是換頁——
    真實換頁實測落在 0.60–0.99。這是 `min_ink_change` 下限的作用。"""
    clean = np.full(100, 0.0)
    clean[50] = 0.05
    assert viterbi_changes(clean, 0.97, 0.01) == []


def test_empty_sequence_yields_no_changes():
    assert viterbi_changes(np.array([]), 0.97, 0.01) == []


def test_bulk_quantile_assumes_changes_are_rare():
    """0.75 等於假設換頁幀不超過四分之一——1fps 下相當於每 4 秒換一頁，
    比任何真實講課都密集得多。"""
    assert 0.5 <= BULK_QUANTILE <= 0.9


def test_self_transition_encodes_sdd_dwell_time():
    """0.97 對應幾何分布平均停留 ~33 幀。SDD §5.1 寫「每頁停留 30–120 秒」。"""
    from weft.config import S1bConfig

    p = S1bConfig()
    mean_dwell = 1.0 / (1.0 - p.hmm_self_transition) / p.fps
    assert 30 <= mean_dwell <= 120


# --------------------------------------------------------------------------
# 逐條動畫合併
# --------------------------------------------------------------------------


def _build_frames() -> list[Frame]:
    """三段逐條動畫（內容遞增），接一段完全不同的頁面。"""
    steps = [
        mask(["##..", "....", "....", "...."]),
        mask(["##..", "##..", "....", "...."]),
        mask(["##..", "##..", "##..", "...."]),
        mask(["....", "....", "....", "..##"]),  # 換頁
    ]
    return [frame(i, ink=steps[i]) for i in range(4)]


def test_progressive_builds_merge_into_one_section():
    frames = _build_frames()
    sections = [Section(i, i + 1, [i]) for i in range(4)]
    merged = merge_progressive(sections, frames, 0.70)

    assert len(merged) == 2
    assert merged[0].is_progressive
    assert merged[0].build_indices == [0, 1, 2]


def test_merged_section_keyframe_is_the_most_complete_frame():
    """SDD §4.3 步驟 5：代表幀取**段內 ink 量最大者**——內容最完整的那一張。
    取到第一幀同樣是 1 張投影片，但內容缺了大半。"""
    frames = _build_frames()
    merged = merge_progressive([Section(i, i + 1, [i]) for i in range(4)], frames, 0.70)
    # 段落只有 3 幀（短於 KEYFRAME_MIN_TRIMMABLE），取正中間；但三個 build
    # 的 ink 是遞增的，所以要驗證的是「不是第一幀」
    assert merged[0].keyframe(frames) > merged[0].start


def test_real_page_change_is_not_merged():
    frames = _build_frames()
    merged = merge_progressive([Section(i, i + 1, [i]) for i in range(4)], frames, 0.70)
    assert not merged[1].is_progressive


def test_merge_is_disabled_cleanly():
    """`progressive_merge=False` 時每段各自獨立，不該悄悄留下半套行為。"""
    frames = _build_frames()
    sections = [Section(i, i + 1, [i]) for i in range(4)]
    assert len(merge_progressive(sections, frames, 1.01)) == 4


def test_containment_threshold_sits_between_measured_populations():
    """0.70 不是隨手挑的：實測 build 的 containment 落在 0.86–0.98，
    真正換頁落在 ≤0.56（見 docs/decisions.md D8）。"""
    from weft.config import S1bConfig

    assert 0.56 < S1bConfig().progressive_containment_ratio < 0.86


# --------------------------------------------------------------------------
# §5.5 #1：不得以固定間隔取樣冒充換頁偵測
# --------------------------------------------------------------------------


def test_detector_output_is_not_uniformly_spaced():
    """機械式護欄：若哪天有人把偵測換成「每 N 秒切一刀」，段落長度會變成
    定值。這裡用一段長度懸殊的距離序列，斷言輸出反映的是內容而非時鐘。"""
    d = np.full(120, 0.003)
    for i in (10, 15, 90):  # 刻意不等距
        d[i] = 0.9
    changes = viterbi_changes(d, 0.97, 0.01)
    gaps = [b - a for a, b in zip(changes, changes[1:])]
    assert changes == [10, 15, 90]
    assert len(set(gaps)) > 1, "段落等距，疑似固定間隔取樣（§5.5 #1）"


# --------------------------------------------------------------------------
# 代表幀選取（SDD §4.3 步驟 5，v0.3 改版）
# --------------------------------------------------------------------------


def _section_frames(inks: list[np.ndarray]) -> list[Frame]:
    return [frame(i, ink=m) for i, m in enumerate(inks)]


def test_keyframe_picks_the_frame_with_most_ink():
    """代表幀取段內 ink 量最大者——逐條動畫的最後一個 build 內容最完整。"""
    inks = [
        mask(["#...", "....", "....", "...."]),
        mask(["##..", "....", "....", "...."]),
        mask(["###.", "#...", "....", "...."]),  # 最多
        mask(["##..", "....", "....", "...."]),
        mask(["#...", "....", "....", "...."]),
        mask(["#...", "....", "....", "...."]),
    ]
    frames = _section_frames(inks)
    assert Section(0, 6, [0]).keyframe(frames) == 2


def test_keyframe_excludes_both_ends():
    """§4.3 步驟 5 的核心：**排除兩端**，避開交叉淡化轉場幀。

    這裡把 ink 量最大的幀刻意放在段落的第一幀——那正是轉場幀會出現的位置
    （混合了兩個畫面，ink 量往往最高）。代表幀不得取到它。
    """
    inks = [mask(["####", "####", "####", "####"])]  # 第一幀 ink 最多（模擬轉場）
    inks += [mask(["##..", "....", "....", "...."])] * 5
    frames = _section_frames(inks)

    picked = Section(0, 6, [0]).keyframe(frames)
    assert picked >= KEYFRAME_EDGE_MARGIN, "代表幀落在段落開頭，可能是轉場幀"
    assert picked < 6 - KEYFRAME_EDGE_MARGIN, "代表幀落在段落結尾，可能是轉場幀"


def test_keyframe_excludes_trailing_transition():
    """段末同樣可能是轉場幀（淡出）。"""
    inks = [mask(["##..", "....", "....", "...."])] * 5
    inks += [mask(["####", "####", "####", "####"])]  # 最後一幀 ink 最多
    frames = _section_frames(inks)
    assert Section(0, 6, [0]).keyframe(frames) < 6 - KEYFRAME_EDGE_MARGIN


def test_short_section_falls_back_to_midpoint():
    """段落短於 KEYFRAME_MIN_TRIMMABLE 時排除兩端會無幀可選，改取正中間。"""
    frames = _section_frames([mask(["#..."])] * 3)
    assert Section(0, 3, [0]).keyframe(frames) == 1


def test_single_frame_section_is_handled():
    frames = _section_frames([mask(["#..."])])
    assert Section(0, 1, [0]).keyframe(frames) == 0


def test_empty_section_raises():
    """空區段是上游的 bug，不該悄悄回傳一個看似合理的 index。"""
    with pytest.raises(ValueError, match="空區段"):
        Section(3, 3, [3]).keyframe(_section_frames([mask(["#..."])] * 4))
