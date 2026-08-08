"""S1c 投影片去重（SDD §4.3b、docs/decisions.md D26）。

案例取自 `zIglvjoU9vo` 的實測：49 個候選幀 → 22 張相異投影片，
其中攝影棚定鏡 17 張聚成一群、《內觀經》10 張、《一月為胞》3 張。
"""

from __future__ import annotations

import numpy as np
import pytest

from weft.config import Config
from weft.ir import Slide
from weft.stages.dedup import group_by_ink, representatives, s1c_dedup


def _mask(pattern: str) -> np.ndarray:
    """用字元圖建遮罩，`#` 為前景。"""
    return np.array([[c == "#" for c in row] for row in pattern.split("|")], dtype=bool)


def _slide(sid: str, t0: float, t1: float) -> Slide:
    return Slide(slide_id=sid, image_path=f"03_slides/{sid}.png",
                 t_first_seen=t0, t_last_seen=t1)


# ---------------------------------------------------------------------------
# 分群本身
# ---------------------------------------------------------------------------

def test_identical_masks_group_together():
    m = _mask("##..|.##.")
    groups = group_by_ink(["a", "b", "c"], {"a": m, "b": m.copy(), "c": _mask("....|....")}, 0.30)
    assert sorted(len(g) for g in groups) == [1, 2]


def test_single_linkage_chains_progressive_builds():
    """單連結是刻意的：逐條動畫的中間幀與最終幀之間是**鏈狀**相似。

    a–b 夠近、b–c 夠近，但 a–c 未必——完全連結會把它們拆成兩群。
    """
    a = _mask("#...|....|....")
    b = _mask("#...|#...|....")
    c = _mask("#...|#...|#...")
    groups = group_by_ink(["a", "b", "c"], {"a": a, "b": b, "c": c}, 0.70)
    assert len(groups) == 1, "單連結應把整條 build 鏈併成一群"


def test_grouping_is_order_stable():
    m = _mask("##..|....")
    ids = ["s3", "s1", "s2"]
    groups = group_by_ink(ids, {i: m.copy() for i in ids}, 0.30)
    assert groups == [["s3", "s1", "s2"]], "輸出順序要跟著輸入，才可重現"


# ---------------------------------------------------------------------------
# 整片套用
# ---------------------------------------------------------------------------

class _FakeWork:
    """讓 s1c_dedup 可以在沒有真實圖檔的情況下測。"""

    def __init__(self, tmp_path, masks):
        self.dir = tmp_path
        self.video_id = "vid"
        self.frames_dir = tmp_path / "02_frames"
        self._masks = masks


@pytest.fixture
def patched(monkeypatch):
    def apply(masks):
        monkeypatch.setattr("weft.stages.dedup._dedup_mask",
                            lambda path, short: masks[str(path).rsplit("/", 1)[-1][:-4]])
        monkeypatch.setattr("weft.stages.dedup.median_frame", lambda *a, **k: None)
    return apply


def test_duplicates_point_at_the_representative(tmp_path, patched):
    same = _mask("###.|###.")
    other = _mask("....|...#")
    patched({"slide_001": same, "slide_002": same.copy(), "slide_003": other})
    slides = [_slide("slide_001", 0, 10), _slide("slide_002", 20, 30), _slide("slide_003", 40, 50)]

    stats = s1c_dedup(Config(), _FakeWork(tmp_path, None), slides, None)

    assert stats["distinct"] == 2
    reps = representatives(slides)
    assert len(reps) == 2
    dup = next(s for s in slides if s.duplicate_of)
    assert dup.duplicate_of in {r.slide_id for r in reps}


def test_occurrences_collect_every_appearance(tmp_path, patched):
    """同一張投影片反覆出現時，代表幀要記下**所有**時段。

    實測《太上老君內觀經》在 42 分鐘內出現 10 次。
    """
    same = _mask("###.|###.")
    masks = {f"slide_{i:03d}": same.copy() for i in (1, 2, 3)}
    masks["slide_004"] = _mask("....|...#")   # 一張相異的，避開 max_group_ratio 守衛
    patched(masks)
    slides = [_slide("slide_001", 0, 10), _slide("slide_002", 100, 110),
              _slide("slide_003", 200, 210), _slide("slide_004", 300, 310)]

    s1c_dedup(Config(), _FakeWork(tmp_path, None), slides, None)

    rep = next(s for s in representatives(slides) if len(s.occurrences) > 1)
    assert rep.occurrences == [(0, 10), (100, 110), (200, 210)]
    assert rep.t_first_seen == 0 and rep.t_last_seen == 210


def test_every_slide_has_occurrences_even_without_dedup(tmp_path, patched):
    """未去重時也要有 `occurrences`——下游可以無條件依賴它。"""
    patched({"slide_001": _mask("#...|....")})
    slides = [_slide("slide_001", 5, 15)]
    s1c_dedup(Config(), _FakeWork(tmp_path, None), slides, None)
    assert slides[0].occurrences == [(5, 15)]
    assert slides[0].duplicate_of is None


def test_merged_candidates_are_not_deleted(tmp_path, patched):
    """§5.6 的人工抽檢要能複核「這兩張真的是同一張嗎」。"""
    same = _mask("###.|###.")
    patched({"slide_001": same, "slide_002": same.copy()})
    slides = [_slide("slide_001", 0, 10), _slide("slide_002", 20, 30)]
    s1c_dedup(Config(), _FakeWork(tmp_path, None), slides, None)
    assert len(slides) == 2, "被合併的候選幀不得從清單移除"


def test_too_loose_threshold_skips_dedup_instead_of_tuning(tmp_path, patched):
    """§4.3b／§5.5 #4：門檻過鬆時**跳過去重**，不自動調參數。"""
    same = _mask("###.|###.")
    ids = [f"slide_{i:03d}" for i in range(1, 6)]
    patched({i: same.copy() for i in ids})
    slides = [_slide(i, n * 10, n * 10 + 5) for n, i in enumerate(ids)]

    stats = s1c_dedup(Config(), _FakeWork(tmp_path, None), slides, None)

    assert stats["skipped"] is not None
    assert all(s.duplicate_of is None for s in slides), "跳過時不得留下半套結果"


def test_disabled_is_a_noop(tmp_path, patched):
    same = _mask("###.|###.")
    patched({"slide_001": same, "slide_002": same.copy()})
    slides = [_slide("slide_001", 0, 10), _slide("slide_002", 20, 30)]
    cfg = Config()
    cfg.s1c.enabled = False
    stats = s1c_dedup(cfg, _FakeWork(tmp_path, None), slides, None)
    assert stats["skipped"] == "未啟用"
    assert all(s.duplicate_of is None for s in slides)


def test_representative_is_the_one_with_most_ink(tmp_path, patched):
    """同 §4.3 步驟 5：取最完整的一幀。"""
    # Jaccard 距離 = 1 − 6/7 ≈ 0.14 < 0.30，會合併；big 的 ink 量較大
    small = _mask("###.|###.")
    big = _mask("####|###.")
    patched({"slide_001": small, "slide_002": big, "slide_003": _mask("....|...#")})
    slides = [_slide("slide_001", 0, 10), _slide("slide_002", 20, 30),
              _slide("slide_003", 40, 50)]
    s1c_dedup(Config(), _FakeWork(tmp_path, None), slides, None)
    assert slides[0].duplicate_of == "slide_002", "代表幀應為 ink 量較大的那張"


def test_guard_fires_when_everything_is_one_group(tmp_path, patched):
    """全部併成一群時守衛會觸發——即使那是**正確的**（整支只有一張投影片）。

    後果只是多送幾次 VLM，不會產出錯的東西。守衛的用意是抓「門檻過鬆」，
    寧可在這種邊界情形上保守。
    """
    same = _mask("###.|###.")
    patched({"slide_001": same, "slide_002": same.copy()})
    slides = [_slide("slide_001", 0, 10), _slide("slide_002", 20, 30)]
    stats = s1c_dedup(Config(), _FakeWork(tmp_path, None), slides, None)
    assert stats["skipped"] is not None
    assert all(s.duplicate_of is None for s in slides)


# ---------------------------------------------------------------------------
# R25：分組必須與**獨立的度量**一致
#
# `slide_groups` 的第一版是從 S1c 自己的輸出抄的——循環論證，
# 100% 是必然的不是成績（SDD §5.1(B) 第 4 條明文禁止）。
# 人工複核只擋得到誤併，擋不到漏併：複核者看的是演算法給的分組，
# 不會被問到「有沒有該分而沒分的」。
# ---------------------------------------------------------------------------

def test_grouping_agrees_with_an_independent_metric():
    """ink Jaccard 的分組，用**原始像素平均絕對差**去驗——兩者不得有重疊。

    受測演算法看前景圖樣（Otsu 遮罩），驗證者看整體亮度分布，機制不同。
    實測三支影片 1,000+ 個配對零分歧。
    """
    import json
    from pathlib import Path

    import cv2
    import numpy as np

    work_root = Path("work")
    checked = 0
    for golden in sorted(Path("tests/golden").glob("*.golden.json")):
        video_id = json.loads(golden.read_text(encoding="utf-8"))["video_id"]
        dedup_path = work_root / video_id / "04_dedup.json"
        if not dedup_path.exists():
            continue

        dedup = json.loads(dedup_path.read_text(encoding="utf-8"))["slides"]
        rep = {s: (info["duplicate_of"] or s) for s, info in dedup.items()}
        sids = sorted(rep)
        tiny = {}
        for sid in sids:
            img = cv2.imread(str(work_root / video_id / "03_slides" / f"{sid}.png"),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            tiny[sid] = cv2.resize(img, (160, 90),
                                   interpolation=cv2.INTER_AREA).astype(np.int16)
        if len(tiny) < 2:
            continue

        merged, apart = [], []
        for i, a in enumerate(sids):
            if a not in tiny:
                continue
            for b in sids[i + 1:]:
                if b not in tiny:
                    continue
                diff = float(np.abs(tiny[a] - tiny[b]).mean()) / 255
                (merged if rep[a] == rep[b] else apart).append(diff)
        if not merged or not apart:
            continue

        checked += 1
        assert min(apart) > max(merged), (
            f"{video_id}：ink Jaccard 的分組與像素差不一致——"
            f"未合併最小 {min(apart):.4f} ≤ 已合併最大 {max(merged):.4f}。"
            f"兩個獨立度量畫不出同一條線時，分組很可能是演算法的產物。"
        )

    if not checked:
        pytest.skip("沒有任何影片同時有黃金集與 04_dedup.json")
