"""端到端測試。SDD §7.1：「空的 e2e 測試（跑不通，但存在）」。

§5.5 #10：**e2e 測試不得用 mock 取代真實模型呼叫。** 這裡沒有任何 mock，
所以在階段實作完成前，這些測試會以 `StageNotImplemented` 失敗——而那正是
Phase 0 的完成條件：「跑 pytest 會失敗，但失敗訊息清楚指出缺少哪些實作」。

各測試的 marker 對應它需要的資源，方便分階段開啟：
    pytest -m "not gpu and not quota"   Phase 0/1 的 CI
    pytest -m synth                     只跑合成影片相關
"""

from __future__ import annotations

from pathlib import Path

import pytest

from weft.config import Config
from weft.paths import OutPaths, WorkPaths
from weft.stages import StageNotImplemented
from weft.validation import invariants as inv
from weft.validation.thresholds import (
    BOUNDARY_F1_SYNTHETIC,
    BOUNDARY_TOLERANCE_SEC,
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
def test_interleaved_speaker_and_slides_are_not_over_segmented(synth_work, cfg: Config):
    """A6：講者頁與投影片頁交錯。

    **v0.3 改變了期望行為。** 舊版量 speaker/slide 分類 accuracy；v0.3 沒有
    分類器了（見 docs/decisions.md D16），改為驗證區段數量合理——每個
    邏輯頁面（3 張投影片 + 2 段講者）各自成段，不該爆量。
    """
    from tests.synth.scenarios import A6
    from weft.stages.local import s1b_slides

    _candidates, slides = s1b_slides(cfg, WorkPaths(synth_work, A6.name))
    # 5 個邏輯頁面；容許 ±2 的偏差（轉場、片頭）
    assert 3 <= len(slides) <= 7, f"A6 切成 {len(slides)} 段，與 5 個邏輯頁面差距過大"


@pytest.mark.synth
@pytest.mark.parametrize(
    "scenario_name",
    ["A1_standard", "A2_progressive", "A3_speaker_only", "A4_laser_pointer",
     "A5_embedded_video", "A6_interleaved", "A7_backtrack"],
)
def test_local_pipeline_satisfies_all_invariants(synth_work, cfg: Config, scenario_name):
    """S1b→S3 的完整本地管線，跑完後 §5.3 的不變量必須全數通過。

    這比逐條單元測試強得多：單元測試用的是手造資料，這裡用的是**真實
    管線輸出**——時間戳來自 ffmpeg 抽幀、segment 來自 HMM、cue 指派來自
    對齊。不變量 1/2/3 的連鎖失敗只有在這種條件下才看得出來。

    v0.3 的本地管線只剩 S1b → S1a → S3，全部不需要 GPU 也不花額度。
    """
    from tests.synth.scenarios import BY_NAME
    from weft.ir import Transcript, VideoIR, VideoMeta
    from weft.stages import local

    truth = BY_NAME[scenario_name]
    work = WorkPaths(synth_work, scenario_name)

    candidates, slides = local.s1b_slides(cfg, work)
    transcript = local.s1a_transcript(cfg, work)
    segments = local.s3_align(cfg, work, transcript, candidates)

    ir = VideoIR(
        meta=VideoMeta.model_validate_json(work.meta.read_text(encoding="utf-8")),
        slides=slides,
        segments=segments,
    )
    violations = inv.check_all(ir, transcript, work.dir)
    assert violations == [], "\n".join(str(v) for v in violations)

    # 覆蓋率與 ground truth 一致
    assert segments[-1].t_end == pytest.approx(truth.duration, abs=1.0)


# --------------------------------------------------------------------------
# 真實影片黃金集（SDD §5.1 B）
#
# 標註檔放在 tests/golden/*.golden.json，影片本身不進版控（§9 版權）。
# 沒有標註檔時這些測試會 skip 而**不是**假通過——skip 會出現在報告裡。
# 製作方式見 tests/golden/annotate.py。
# --------------------------------------------------------------------------


def _golden_annotations():
    from tests.golden.annotate import GoldenAnnotation

    root = Path(__file__).resolve().parent / "golden"
    out = []
    for path in sorted(root.glob("*.golden.json")):
        annotation = GoldenAnnotation.load(path)
        if annotation.reviewed:
            out.append(annotation)
    return out


@pytest.mark.golden
def test_golden_set_has_three_videos():
    """§5.1（B）：手工標註 **3 支**真實影片（含目標 playlist 的至少 2 支）。"""
    annotations = _golden_annotations()
    if not annotations:
        pytest.skip("尚無經人工確認的黃金集標註（見 tests/golden/annotate.py）")
    assert len(annotations) >= 3, (
        f"黃金集只有 {len(annotations)} 支，SDD §5.1（B）要求 3 支。"
        "§5.5 #8：不得縮小測試集。"
    )


@pytest.mark.golden
def test_boundary_f1_on_real_videos(cfg: Config):
    """§5.2：真實影片的換頁偵測 boundary F1 ≥ 0.75。

    參考：文獻中 naive frame diff 約 0.50–0.60，好方法約 0.81+。
    §5.5 #7：**不得為了讓測試通過而調低此門檻。**
    """
    from weft.stages.local import s1b_slides
    from weft.validation.metrics import boundary_prf
    from weft.validation.thresholds import BOUNDARY_F1_REAL

    annotations = _golden_annotations()
    if not annotations:
        pytest.skip("尚無經人工確認的黃金集標註（見 tests/golden/annotate.py）")

    failures = []
    for annotation in annotations:
        work = WorkPaths(cfg.work_dir, annotation.video_id)
        if not work.video.exists():
            pytest.skip(f"{annotation.video_id} 的影片不在 work/（請先 weft prepare）")
        candidates, _slides = s1b_slides(cfg, work)
        predicted = _internal_boundaries(candidates, annotation.duration)
        prf = boundary_prf(predicted, annotation.confirmed, BOUNDARY_TOLERANCE_SEC)
        if prf.f1 < BOUNDARY_F1_REAL:
            failures.append(f"{annotation.video_id}: {prf}")
    assert not failures, "\n".join(failures)


@pytest.mark.golden
def test_frame_classification_on_real_videos(cfg: Config):
    """§5.2：speaker/slide 分類 accuracy ≥ 0.95（真實影片）。"""
    from weft.stages.local import s1b_slides
    from weft.validation.metrics import classification_accuracy
    from weft.validation.thresholds import FRAME_CLASS_ACCURACY

    annotations = [a for a in _golden_annotations() if a.frame_classes]
    if not annotations:
        pytest.skip("黃金集尚未標註 frame_classes")

    for annotation in annotations:
        work = WorkPaths(cfg.work_dir, annotation.video_id)
        candidates, _ = s1b_slides(cfg, work)
        by_time = {round(f.t, 1): str(f.frame_class) for f in candidates.frames}
        pairs = [
            (by_time[round(float(t), 1)], label)
            for t, label in annotation.frame_classes.items()
            if round(float(t), 1) in by_time
        ]
        assert pairs, f"{annotation.video_id}：標註的時間點與抽幀對不上"
        accuracy = classification_accuracy([p for p, _ in pairs], [g for _, g in pairs])
        assert accuracy >= FRAME_CLASS_ACCURACY, f"{annotation.video_id}: {accuracy:.3f}"


@pytest.mark.golden
@pytest.mark.gpu
def test_alignment_boundary_error_within_threshold(cfg: Config):
    """§5.2：對齊邊界誤差中位數 ≤ 5 秒（黃金集）。"""
    pytest.skip("需要黃金集標註 + 完整 prepare 產物；標註完成後啟用")


@pytest.mark.golden
@pytest.mark.gpu
def test_term_correction_precision(cfg: Config):
    """§5.2：術語校正 precision ≥ 0.90（黃金集）。寧可漏改，不可亂改。

    合成素材版見 `test_term_correction_precision_on_synthetic`——兩者互補：
    黃金集測真實 Whisper／字幕的錯誤分布，合成版測演算法在已知錯誤下的行為。
    """
    pytest.skip("需要黃金集標註逐字稿中的術語錯誤；標註完成後啟用")


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


def test_no_stage_is_left_unimplemented():
    """SDD §4 的十個階段全部有實作。

    這條測試取代了 Phase 0 的「未實作階段須報出缺什麼」——那條的任務
    （§7.1：「跑 pytest 會失敗，但失敗訊息清楚指出缺少哪些實作」）已經
    完成，S0–S6 全部落地。現在改為反向守則：**不得有階段悄悄退回 stub**。

    `pending()` 的機制保留，供未來新增階段時沿用。
    """
    import ast
    from pathlib import Path

    from weft.state import Stage

    src = Path(__file__).resolve().parents[1] / "src" / "weft" / "stages"
    stubbed: list[str] = []
    for module in ("local.py", "cloud.py"):
        tree = ast.parse((src / module).read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "pending"
                ):
                    stubbed.append(f"{module}:{node.name}")
    assert not stubbed, f"這些階段仍是 stub：{stubbed}"

    # 每個 SDD 階段都要有對應的參數 hash（否則續跑判斷會漏掉它）
    from weft.config import Config
    from weft.pipeline import stage_params

    cfg = Config()
    assert len({stage_params(cfg, s) for s in Stage}) == len(Stage)


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
