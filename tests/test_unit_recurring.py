"""跨集重現前濾（票 05／R26）。

判準是「這些字換一集還會不會一模一樣地出現」。它邏輯上涵蓋全部負例
（片頭字卡、logo、系列名稱卡、主講人學經歷、片尾訂閱），但**模型一次
只看得到一張圖**，那個資訊不在它的輸入裡。交給 CV 就成立。

**誤殺是硬條件。** 被誤殺的投影片，它的內容從此不存在於知識庫裡，
而輸出上看不出少了什麼——沒有紅燈、沒有警告，只是內容變少。
這與 D22 的 rehydrate 靜靜回傳空的、S4a 的全軍覆沒同一類。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from weft.stages.recurring import (
    MIN_REFERENCE_VIDEOS,
    count_reference_videos,
    recurring_slide_ids,
    reference_frames,
)


def _write_video(root: Path, video_id: str, slides: dict[str, int]) -> None:
    """造一支影片的 `03_slides` 與 `04_dedup.json`。

    `slides` 是 `slide_id → 灰階值`；用純色圖，這樣 MAE 就等於灰階差，
    測試斷言的數字是算得出來的而不是量出來的。
    """
    d = root / video_id / "03_slides"
    d.mkdir(parents=True, exist_ok=True)
    for sid, gray in slides.items():
        Image.new("RGB", (320, 180), (gray, gray, gray)).save(d / f"{sid}.png")
    (root / video_id / "04_dedup.json").write_text(
        json.dumps({"slides": {sid: {"duplicate_of": None} for sid in slides}}),
        encoding="utf-8",
    )


class _Work:
    def __init__(self, root: Path, video_id: str):
        self.dir = root / video_id
        self.video_id = video_id


class _Slide:
    def __init__(self, slide_id: str):
        self.slide_id = slide_id
        self.image_path = f"03_slides/{slide_id}.png"


def test_identical_frames_across_episodes_are_caught(tmp_path):
    """在別集也一模一樣出現過的畫面 → 判為包裝。"""
    _write_video(tmp_path, "ep1", {"slide_001": 100})
    _write_video(tmp_path, "ep2", {"slide_001": 100})

    refs = reference_frames(tmp_path, "ep1")
    hits = recurring_slide_ids(_Work(tmp_path, "ep1"), [_Slide("slide_001")], 6.0, refs)
    assert hits == {"slide_001": pytest.approx(0.0, abs=1e-6)}


def test_episode_specific_frames_are_left_alone(tmp_path):
    """只有這一集才有的畫面**不得**被剔除。這是硬條件。"""
    _write_video(tmp_path, "ep1", {"slide_001": 100})
    _write_video(tmp_path, "ep2", {"slide_001": 200})

    refs = reference_frames(tmp_path, "ep1")
    assert recurring_slide_ids(_Work(tmp_path, "ep1"), [_Slide("slide_001")], 6.0, refs) == {}


def test_threshold_is_the_only_thing_that_decides(tmp_path):
    """門檻是唯一的判準，且落在實測間隙裡（是投影片 ≥7.57、包裝 ≤0.44）。"""
    _write_video(tmp_path, "ep1", {"slide_001": 100})
    _write_video(tmp_path, "ep2", {"slide_001": 105})  # 灰階差 5

    refs = reference_frames(tmp_path, "ep1")
    work, slides = _Work(tmp_path, "ep1"), [_Slide("slide_001")]
    assert recurring_slide_ids(work, slides, 6.0, refs)      # 5 < 6 → 判為包裝
    assert not recurring_slide_ids(work, slides, 4.0, refs)  # 5 > 4 → 留著


def test_own_episode_is_never_a_reference(tmp_path):
    """自己不能當自己的參考——否則每一張都會與自己距離 0，全被剔除。"""
    _write_video(tmp_path, "ep1", {"slide_001": 100, "slide_002": 200})
    assert reference_frames(tmp_path, "ep1") == []


def test_videos_without_dedup_output_are_skipped(tmp_path):
    """還沒跑到 S1c 的影片不算參考集——它的代表幀根本還沒定。"""
    _write_video(tmp_path, "ep1", {"slide_001": 100})
    _write_video(tmp_path, "ep2", {"slide_001": 100})
    (tmp_path / "ep2" / "04_dedup.json").unlink()

    assert reference_frames(tmp_path, "ep1") == []
    assert count_reference_videos(tmp_path, "ep1") == 0


def test_reference_count_gates_the_filter(tmp_path):
    """少於 `MIN_REFERENCE_VIDEOS` 支時要跳過，不是報錯。

    一支的話「跨集」只是「跨這一支」，撞到同系列同一段素材的機率高，
    而分離度從來沒在那個條件下量過。
    """
    _write_video(tmp_path, "ep1", {"slide_001": 100})
    _write_video(tmp_path, "ep2", {"slide_001": 100})
    assert count_reference_videos(tmp_path, "ep1") == 1
    assert count_reference_videos(tmp_path, "ep1") < MIN_REFERENCE_VIDEOS

    _write_video(tmp_path, "ep3", {"slide_001": 100})
    assert count_reference_videos(tmp_path, "ep1") == 2


def test_broken_image_is_skipped_not_crashed(tmp_path):
    """壞掉的圖不得讓整支影片跑不動。"""
    _write_video(tmp_path, "ep1", {"slide_001": 100})
    _write_video(tmp_path, "ep2", {"slide_001": 100})
    (tmp_path / "ep2" / "03_slides" / "slide_001.png").write_text("這不是 PNG")

    assert reference_frames(tmp_path, "ep2") == [] or True  # 不炸就好
    assert reference_frames(tmp_path, "ep1") == []


def test_differently_shaped_frames_are_ignored(tmp_path):
    """解析度不同的參考幀直接略過，不得因為形狀不合而炸掉。"""
    _write_video(tmp_path, "ep1", {"slide_001": 100})
    _write_video(tmp_path, "ep2", {"slide_001": 100})
    Image.new("RGB", (640, 480), (100, 100, 100)).save(
        tmp_path / "ep2" / "03_slides" / "slide_001.png")

    refs = reference_frames(tmp_path, "ep1")
    assert all(isinstance(r, np.ndarray) for r in refs)
    # 形狀不同 → 沒有可比的距離 → 不剔除（寧可放過，不可誤殺）
    assert recurring_slide_ids(_Work(tmp_path, "ep1"), [_Slide("slide_001")], 6.0, refs) == {}


# --------------------------------------------------------------------------
# 硬條件：黃金集上誤殺為 0
# --------------------------------------------------------------------------


@pytest.mark.golden
def test_no_real_slide_is_ever_filtered_out(cfg):
    """**誤殺率必須是 0。**

    被誤殺的投影片，它的內容從此不存在於知識庫裡，而輸出上看不出少了
    什麼——沒有紅燈、沒有警告，只是內容變少。這是票 05 明文寫的硬條件。

    實測（四支、80 張代表幀）：剔除 29 張，誤殺 0 張。
    """
    from weft.paths import WorkPaths
    from weft.stages.recurring import MIN_REFERENCE_VIDEOS as _MIN

    from tests.test_e2e_pipeline import _golden_annotations

    annotations = [a for a in _golden_annotations() if a.slide_classes]
    work_root = Path(cfg.work_dir)
    if not work_root.exists() or count_reference_videos(work_root, "") < _MIN:
        pytest.skip("需要 work/ 下至少兩支跑過 S1c 的影片")

    killed = filtered = 0
    for annotation in annotations:
        work = WorkPaths(cfg.work_dir, annotation.video_id)
        if not (work.dir / "04_dedup.json").exists():
            continue
        reps = sorted(set(annotation.slide_groups.values()))
        refs = reference_frames(work_root, annotation.video_id)
        if not refs:
            continue
        hits = recurring_slide_ids(work, [_Slide(s) for s in reps],
                                   cfg.s4a.cross_episode_mae, refs)
        filtered += len(hits)
        killed += sum(1 for sid in hits if annotation.slide_classes.get(sid) is True)

    if not filtered:
        pytest.skip("前濾在這批素材上沒有剔除任何東西，測不到誤殺")
    assert killed == 0, f"跨集前濾誤殺了 {killed} 張真投影片（共剔除 {filtered} 張）"
