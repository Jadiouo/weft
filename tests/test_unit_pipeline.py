"""`prepare_one` 的失敗歸屬（2026-08-09）。

**錯誤訊息指錯地方比沒有錯誤訊息更糟**：它會主動把人帶往錯的方向。

實測：Whisper 的 `CUDA failed with error out of memory`（S1a）被記成
「S0 取得失敗」，因為 `run_prepare` 的例外處理寫死了 `Stage.S0_FETCH`。
查問題的人會去看下載，而下載完全正常——影片就在 `01_video.mp4`。
D29（規約檢查的行號一直是錯的）是同一類。

**全部階段都要 stub**，包括沒在測的那些：只 stub 被測的那一個，
其餘會真的去打網路，而預設測試層必須是離線的（票 02）。
"""

from __future__ import annotations

import json

import pytest

from weft.config import Config
from weft.ir import CandidateSet, Transcript, TranscriptCue, TranscriptSource, VideoMeta
from weft.paths import WorkPaths
from weft.pipeline import StageFailure, run_prepare
from weft.state import Stage, StageStatus, VideoState

#: 合法的 YouTube id 形狀（11 字元）——`parse_target` 只認這個長度。
_VID = "aaaaaaaaaaa"


def _cfg(tmp_path):
    cfg = Config()
    cfg.work_dir = tmp_path / "work"
    cfg.out_dir = tmp_path / "out"
    # S-1 會抽幀，與本檔要測的東西無關
    cfg.survey_each_video = False
    return cfg


def _stub_all(monkeypatch, cfg):
    """把 S0–S3 全部換成不碰外部資源的假實作。"""
    meta = VideoMeta(video_id=_VID, title="t", duration=90.0,
                     url=f"https://example.invalid/{_VID}", video_path="01_video.mp4")

    def fake_fetch(video_id, _cfg, work):
        work.meta.write_text(meta.model_dump_json(), encoding="utf-8")
        return meta

    def fake_slides(_cfg, work):
        cands = CandidateSet(video_id=_VID, fps=1.0, duration=90.0,
                             frames=[], candidates=[], params_hash="x")
        work.candidates.write_text(cands.model_dump_json(), encoding="utf-8")
        return cands, []

    def fake_transcript(_cfg, work):
        cues = [TranscriptCue(index=0, t_start=0.0, t_end=90.0, text_raw="內容")]
        tr = Transcript(video_id=_VID, source=TranscriptSource.WHISPER, cues=cues,
                        raw_hash=Transcript.compute_raw_hash(cues))
        work.transcript.write_text(tr.model_dump_json(), encoding="utf-8")
        return tr

    monkeypatch.setattr("weft.stages.local.s0_fetch", fake_fetch)
    monkeypatch.setattr("weft.stages.local.s1b_slides", fake_slides)
    monkeypatch.setattr("weft.stages.local.s1a_transcript", fake_transcript)
    monkeypatch.setattr("weft.stages.dedup.s1c_dedup", lambda *_a, **_k: {})
    monkeypatch.setattr("weft.stages.local.s3_align", lambda *_a, **_k: [])


@pytest.mark.parametrize(
    ("stage", "target"),
    [
        (Stage.S0_FETCH, "weft.stages.local.s0_fetch"),
        (Stage.S1B_SLIDES, "weft.stages.local.s1b_slides"),
        (Stage.S1C_DEDUP, "weft.stages.dedup.s1c_dedup"),
        (Stage.S1A_TRANSCRIPT, "weft.stages.local.s1a_transcript"),
        (Stage.S3_ALIGN, "weft.stages.local.s3_align"),
    ],
)
def test_failure_is_recorded_against_the_stage_that_failed(
    tmp_path, monkeypatch, stage, target
):
    """哪個階段爆掉就記在哪個階段，不是一律記 S0。"""
    cfg = _cfg(tmp_path)
    _stub_all(monkeypatch, cfg)

    def explode(*_a, **_k):
        raise RuntimeError("CUDA failed with error out of memory")

    monkeypatch.setattr(target, explode)
    run_prepare(_VID, cfg)

    state = VideoState.load_or_new(WorkPaths(cfg.work_dir, _VID).state, _VID)
    record = state.stages[stage.value]
    assert record.status is StageStatus.FAILED, (
        f"{stage.value} 失敗卻沒有記在 {stage.value} 上"
    )
    assert "out of memory" in (record.error or ""), "原始錯誤訊息不得被吃掉"

    others = [s for s, r in state.stages.items()
              if s != stage.value and r.status is StageStatus.FAILED]
    assert not others, f"這些階段被誤標為失敗：{others}"


def test_stage_failure_keeps_the_original_cause():
    """包一層不能把原始例外弄丟——那是查問題唯一的線索。"""
    cause = ValueError("原始訊息")
    wrapped = StageFailure(Stage.S1A_TRANSCRIPT, cause)
    assert wrapped.stage is Stage.S1A_TRANSCRIPT
    assert wrapped.cause is cause
    assert "原始訊息" in str(wrapped)
    assert "S1a" in str(wrapped)


def test_video_unavailable_is_not_wrapped(tmp_path, monkeypatch):
    """`VideoUnavailable` 要原樣往上——§4.1 對它有專門處置（記進 skip list，
    不算成失敗）。包成 `StageFailure` 會讓那條路走不到。"""
    from weft.stages.fetch import VideoUnavailable

    cfg = _cfg(tmp_path)
    _stub_all(monkeypatch, cfg)

    def unavailable(*_a, **_k):
        raise VideoUnavailable("私人影片")

    monkeypatch.setattr("weft.stages.local.s0_fetch", unavailable)
    assert run_prepare(_VID, cfg) == 0, "不可用的影片不算失敗"

    skips = json.loads((cfg.out_dir / "skiplist.json").read_text(encoding="utf-8"))
    assert _VID in skips


def test_success_leaves_no_failed_stage(tmp_path, monkeypatch):
    """沒有這一條，上面那些可以靠「把所有階段都標成失敗」通過。"""
    cfg = _cfg(tmp_path)
    _stub_all(monkeypatch, cfg)
    assert run_prepare(_VID, cfg) == 0

    state = VideoState.load_or_new(WorkPaths(cfg.work_dir, _VID).state, _VID)
    assert not [s for s, r in state.stages.items() if r.status is StageStatus.FAILED]
