"""端到端測試。SDD §7.1：「空的 e2e 測試（跑不通，但存在）」。

§5.5 #10：**e2e 測試不得用 mock 取代真實模型呼叫。** 這裡沒有任何 mock，
所以在階段實作完成前，這些測試會以 `StageNotImplemented` 失敗——而那正是
Phase 0 的完成條件：「跑 pytest 會失敗，但失敗訊息清楚指出缺少哪些實作」。

各測試的 marker 對應它需要的資源，方便分階段開啟：
    pytest -m "not gpu and not quota"   Phase 0/1 的 CI
    pytest -m synth                     只跑合成影片相關
"""

from __future__ import annotations

import pytest

from weft.config import Config
from weft.paths import OutPaths, WorkPaths
from weft.stages import StageNotImplemented
from weft.validation import invariants as inv
from weft.validation.thresholds import (
    BOUNDARY_F1_SYNTHETIC,
    BOUNDARY_TOLERANCE_SEC,
    FRAME_CLASS_ACCURACY,
    PROGRESSIVE_MERGE_ACCURACY,
    PROVENANCE_PASS_RATE,
)

pytestmark = pytest.mark.e2e


# --------------------------------------------------------------------------
# Phase 1：本地管線（S0–S3）
# --------------------------------------------------------------------------


@pytest.mark.synth
@pytest.mark.parametrize(
    "scenario_name",
    ["A1_standard", "A2_progressive", "A3_speaker_only", "A4_laser_pointer",
     "A5_embedded_video", "A6_interleaved", "A7_backtrack"],
)
def test_slide_detection_meets_synthetic_f1_threshold(synth_work, cfg: Config, scenario_name):
    """§5.2：合成影片的換頁偵測 boundary F1 ≥ 0.95（容忍 ±2 秒）。

    §5.5 #7：**不得為了讓測試通過而調低此門檻。**
    """
    from tests.synth.scenarios import BY_NAME
    from weft.stages.local import s1b_slides
    from weft.validation.metrics import boundary_prf

    truth = BY_NAME[scenario_name]
    candidates, _slides = s1b_slides(cfg, WorkPaths(synth_work, scenario_name))

    predicted = _internal_boundaries(candidates, truth.duration)
    prf = boundary_prf(predicted, truth.slide_boundaries, BOUNDARY_TOLERANCE_SEC)
    assert prf.f1 >= BOUNDARY_F1_SYNTHETIC, (
        f"{scenario_name}：{prf}\n  pred={predicted}\n  gt  ={truth.slide_boundaries}"
    )


@pytest.mark.synth
def test_progressive_animation_merges_into_one_slide(synth_work, cfg: Config):
    """§5.2：逐條動畫合併正確率 = 1.00（對抗樣本 A2）。這是設計目標，不容失敗。"""
    from tests.synth.scenarios import A2
    from weft.stages.local import s1b_slides
    from weft.validation.metrics import merge_accuracy

    _candidates, slides = s1b_slides(cfg, WorkPaths(synth_work, A2.name))

    # 每個 slide 邏輯頁面實際被偵測成幾張
    detected = _count_slides_per_page(slides, A2)
    assert merge_accuracy(detected, A2.expected_merge_counts) >= PROGRESSIVE_MERGE_ACCURACY


@pytest.mark.synth
def test_progressive_keyframe_is_the_most_complete_frame(synth_work, cfg: Config):
    """§4.3 步驟 5：逐條動畫取**最後一幀**（內容最完整）。

    只驗證張數是不夠的——取到第一幀同樣是 1 張，但內容缺了 5/6。
    """
    from tests.synth.scenarios import A2
    from weft.stages.local import s1b_slides

    _candidates, slides = s1b_slides(cfg, WorkPaths(synth_work, A2.name))

    build = next(p for p in A2.placed if p.page.build_offsets)
    window = build.keyframe_window
    chosen = [s for s in slides if build.t_start <= s.t_first_seen < build.t_end]
    assert len(chosen) == 1
    assert window[0] <= chosen[0].t_last_seen <= window[1]
    assert chosen[0].is_progressive_final is True


@pytest.mark.synth
def test_speaker_slide_classification_accuracy(synth_work, cfg: Config):
    """§5.2：speaker/slide 分類 accuracy ≥ 0.95。"""
    from tests.synth.scenarios import A6
    from weft.stages.local import s1b_slides
    from weft.validation.metrics import classification_accuracy

    candidates, _slides = s1b_slides(cfg, WorkPaths(synth_work, A6.name))

    predicted = [str(f.frame_class) for f in candidates.frames]
    assert classification_accuracy(predicted, A6.frame_classes()) >= FRAME_CLASS_ACCURACY


@pytest.mark.synth
def test_speaker_only_video_yields_zero_slides(synth_work, cfg: Config):
    """A3：純講者必須偵測為 0 頁，且**不中斷**——退化為 mode=transcript_only（§4.3）。"""
    from tests.synth.scenarios import A3
    from weft.stages.local import s1b_slides

    _candidates, slides = s1b_slides(cfg, WorkPaths(synth_work, A3.name))
    assert slides == []


@pytest.mark.gpu
def test_alignment_boundary_error_within_threshold(cfg: Config):
    """§5.2：對齊邊界誤差中位數 ≤ 5 秒（黃金集）。"""
    pytest.skip("需要真實影片黃金集（SDD §5.1 B），標註工作屬 Phase 1")


@pytest.mark.gpu
def test_term_correction_precision(cfg: Config):
    """§5.2：術語校正 precision ≥ 0.90。寧可漏改，不可亂改。"""
    pytest.skip("需要真實影片黃金集（SDD §5.1 B），標註工作屬 Phase 1")


# --------------------------------------------------------------------------
# Phase 2：理解（S4–S5）
# --------------------------------------------------------------------------


@pytest.mark.quota
@pytest.mark.gpu
def test_single_video_end_to_end_produces_ir(cfg: Config, tmp_path):
    """§7.3 Phase 2 完成條件：單支影片端到端產出 08_video.json，
    溯源通過率 ≥ 0.95。

    §5.5 #10：此測試不得用 mock 取代真實模型呼叫。
    """
    from weft.pipeline import run_prepare, run_understand
    from weft.validation.provenance import check_video

    cfg.work_dir = tmp_path / "work"
    cfg.out_dir = tmp_path / "out"

    video_id = "zIglvjoU9vo"  # SDD §1.3 的代表素材
    assert run_prepare(video_id, cfg) == 0
    assert run_understand(cfg, video_id=video_id) == 0

    work = WorkPaths(cfg.work_dir, video_id)
    assert work.video_ir.exists()

    from weft.ir import Transcript, VideoIR

    ir = VideoIR.model_validate_json(work.video_ir.read_text(encoding="utf-8"))
    transcript = Transcript.model_validate_json(work.transcript.read_text(encoding="utf-8"))

    # §5.3：任一不變量失敗即中止
    inv.assert_all(ir, transcript, work.dir)

    # §5.4：溯源通過率
    assert check_video(ir, cfg.provenance).pass_rate >= PROVENANCE_PASS_RATE


@pytest.mark.quota
def test_quota_exhaustion_stops_without_silent_fallback(cfg: Config, tmp_path):
    """§5.5 #6：額度耗盡時不得靜默改用本地模型。"""
    pytest.skip("需 quota ledger（Phase 2）")


# --------------------------------------------------------------------------
# Phase 3：輸出（S6）
# --------------------------------------------------------------------------


@pytest.mark.quota
@pytest.mark.gpu
def test_chunks_jsonl_passes_all_invariants(cfg: Config, tmp_path):
    """§7.4 Phase 3 完成條件：產出 chunks.jsonl，且 §5.3 全數通過。"""
    from weft.ir import Chunk, Transcript, VideoIR
    from weft.pipeline import run_prepare, run_understand

    cfg.work_dir = tmp_path / "work"
    cfg.out_dir = tmp_path / "out"
    video_id = "zIglvjoU9vo"

    run_prepare(video_id, cfg)
    run_understand(cfg, video_id=video_id)

    out = OutPaths(cfg.out_dir)
    assert out.chunks.exists()

    chunks = [
        Chunk.model_validate_json(line)
        for line in out.chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert chunks, "chunks.jsonl 為空"

    work = WorkPaths(cfg.work_dir, video_id)
    ir = VideoIR.model_validate_json(work.video_ir.read_text(encoding="utf-8"))
    transcript = Transcript.model_validate_json(work.transcript.read_text(encoding="utf-8"))
    inv.assert_all(ir, transcript, work.dir, chunks=chunks)


@pytest.mark.quota
@pytest.mark.gpu
def test_needs_review_videos_do_not_reach_chunks(cfg: Config, tmp_path):
    """§5.4：unverified 比例 > 5% 的影片不得進入 chunks.jsonl。"""
    pytest.skip("需 S4–S6（Phase 2–3）")


# --------------------------------------------------------------------------
# Phase 0 的完成條件
# --------------------------------------------------------------------------


def test_unimplemented_stages_report_what_is_missing(cfg: Config, tmp_path):
    """SDD §7.1 完成條件：「跑 pytest 會失敗，但失敗訊息清楚指出缺少哪些實作」。

    這條測試把該條件變成可驗證的東西：每個未實作階段都必須說明自己是誰、
    對應 SDD 哪一節、屬哪個 Phase、還缺什麼。
    """
    from weft.stages import cloud, local

    work = WorkPaths(tmp_path, "dummy")
    # 已實作的階段從這裡移除——清單本身就是進度表。
    # 已完成：S1b（§4.3，Phase 1）
    cases = [
        ("S0", lambda: local.s0_fetch("v", cfg, work), "§4.1"),
        ("S1a", lambda: local.s1a_transcript(cfg, work), "§4.2"),
        ("S2", lambda: local.s2_ocr(cfg, work, []), "§4.4"),
        ("S2b", lambda: local.s2b_lexicon(cfg, work, [], None), "§4.4"),
        ("S2c", lambda: local.s2c_correct(cfg, work, None, None), "§4.5"),
        ("S3", lambda: local.s3_align(cfg, work, None, None), "§4.6"),
        ("S4", lambda: cloud.s4_understand(cfg, work, [], None), "§4.7"),
        ("S5", lambda: cloud.s5_synthesize(cfg, work, None), "§4.8"),
        ("S6", lambda: cloud.s6_render(cfg, None, work, OutPaths(tmp_path)), "§4.9"),
    ]

    for stage, call, section in cases:
        with pytest.raises(StageNotImplemented) as excinfo:
            call()
        message = str(excinfo.value)
        assert stage in message, f"{stage} 的錯誤訊息未指出階段名稱"
        assert section in message, f"{stage} 的錯誤訊息未指出 SDD 章節"
        assert "Phase" in message, f"{stage} 的錯誤訊息未指出所屬 Phase"
        assert "待實作" in message, f"{stage} 的錯誤訊息未列出待實作項目"


def _internal_boundaries(candidates, duration: float) -> list[float]:
    """從候選段落推導出「投影片段落的起訖時刻」，去掉影片頭尾。

    與 SynthTruth.slide_boundaries 的定義必須一致——slide↔speaker 的切換
    同樣算邊界（A6 的四個邊界全部來自這裡）。
    """
    marks = set()
    for c in candidates.candidates:
        marks.add(round(c.t_start, 3))
        marks.add(round(c.t_end, 3))
    return sorted(m for m in marks if 0.5 < m < duration - 0.5)


def _count_slides_per_page(slides, truth) -> list[int]:
    """每個 slide 邏輯頁面實際被偵測成幾張。"""
    counts = []
    for placed in truth.placed:
        if placed.page.kind != "slide":
            continue
        counts.append(sum(1 for s in slides if placed.t_start <= s.t_first_seen < placed.t_end))
    return counts
