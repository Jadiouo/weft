"""整批 prepare 的回報要誠實（2026-09-01）。

**無人執行的路第一次被走，第一個發現就是它會說謊。**

實測：機器人學播放清單 26 支，23 支下載 403 → 全部走 `VideoUnavailable`
→ 記進 skip list。而舊版只數 `failed`，skip 不算，於是收尾印出：

    prepare 完成：26 支目標，全部成功

第二個洞同樣嚴重：已在 skip list 裡的影片在 `try` 之前就 `continue`，
`failed` 與 `succeeded` 都不計——**重跑一次會什麼都不做，然後再說一次
「全部成功」**。

這兩個洞的共同形狀與這個 repo 抓過的其他東西一樣（門檻只 assert 常數、
`>/dev/null` 吞錯誤、量測讀到舊檔），但這次是**在使用者會直接看到的那一行**。
"""

from __future__ import annotations

import json

import pytest

from weft.config import Config
from weft.ir import CandidateSet, Transcript, TranscriptCue, TranscriptSource, VideoMeta
from weft.pipeline import run_prepare

_VIDS = ("aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc")


def _cfg(tmp_path):
    cfg = Config()
    cfg.work_dir = tmp_path / "work"
    cfg.out_dir = tmp_path / "out"
    cfg.survey_each_video = False
    return cfg


def _stub_stages(monkeypatch):
    """S1–S3 換成不碰外部資源的假實作；S0 由各測試自己決定。"""
    def fake_slides(_cfg, work):
        cands = CandidateSet(video_id=work.video_id, fps=1.0, duration=90.0,
                             frames=[], candidates=[], params_hash="x")
        work.candidates.write_text(cands.model_dump_json(), encoding="utf-8")
        return cands, []

    def fake_transcript(_cfg, work):
        cues = [TranscriptCue(index=0, t_start=0.0, t_end=90.0, text_raw="內容")]
        tr = Transcript(video_id=work.video_id, source=TranscriptSource.WHISPER,
                        cues=cues, raw_hash=Transcript.compute_raw_hash(cues))
        work.transcript.write_text(tr.model_dump_json(), encoding="utf-8")
        return tr

    monkeypatch.setattr("weft.stages.local.s1b_slides", fake_slides)
    monkeypatch.setattr("weft.stages.local.s1a_transcript", fake_transcript)
    monkeypatch.setattr("weft.stages.dedup.s1c_dedup", lambda *_a, **_k: {})
    monkeypatch.setattr("weft.stages.local.s3_align", lambda *_a, **_k: [])
    monkeypatch.setattr("weft.pipeline.resolve_targets",
                        lambda _t: [(v, None, None) for v in _VIDS])


def _ok_fetch(video_id, _cfg, work):
    meta = VideoMeta(video_id=video_id, title="t", duration=90.0,
                     url=f"https://example.invalid/{video_id}",
                     video_path="01_video.mp4")
    work.meta.write_text(meta.model_dump_json(), encoding="utf-8")
    return meta


def _unavailable(video_id, _cfg, work):
    from weft.stages.fetch import VideoUnavailable

    raise VideoUnavailable("HTTP Error 403: Forbidden")


class TestNothingSucceededIsNotSuccess:
    """**這是 2026-09-01 那個 bug 的直接反例。**"""

    def test_all_unavailable_does_not_report_success(self, tmp_path, monkeypatch,
                                                     caplog):
        cfg = _cfg(tmp_path)
        _stub_stages(monkeypatch)
        monkeypatch.setattr("weft.stages.local.s0_fetch", _unavailable)

        with caplog.at_level("INFO"):
            rc = run_prepare("playlist", cfg)

        assert rc == 1, "三支全部不可用，卻回報成功"
        assert "全部成功" not in caplog.text, "收尾訊息裡不得出現「全部成功」"
        assert "沒有任何一支成功" in caplog.text, caplog.text[-300:]
        assert "0 成功、3 不可用" in caplog.text, "拆解的數字要看得見"

    def test_rerun_over_a_full_skiplist_also_fails(self, tmp_path, monkeypatch):
        """**第二個洞**：全部已在 skip list 時重跑，什麼都不做。

        舊版兩邊都不計，於是再說一次「全部成功」。
        """
        cfg = _cfg(tmp_path)
        _stub_stages(monkeypatch)
        monkeypatch.setattr("weft.stages.local.s0_fetch", _unavailable)
        assert run_prepare("playlist", cfg) == 1

        # 第二次：全部已在 skip list，`s0_fetch` 根本不會被呼叫
        called = []
        monkeypatch.setattr("weft.stages.local.s0_fetch",
                            lambda v, c, w: called.append(v))
        assert run_prepare("playlist", cfg) == 1, "重跑什麼都沒做，卻回報成功"
        assert called == [], "已在 skip list 的不該重試（前提檢查）"


class TestTheBreakdownIsAlwaysReported:
    def test_full_success_still_prints_the_numbers(self, tmp_path, monkeypatch,
                                                   caplog):
        """**全成功時也要印數字。** 只說「全部成功」的話，
        讀的人無從分辨「3 支全成功」與「26 支只跑到 3 支」。"""
        cfg = _cfg(tmp_path)
        _stub_stages(monkeypatch)
        monkeypatch.setattr("weft.stages.local.s0_fetch", _ok_fetch)

        with caplog.at_level("INFO"):
            assert run_prepare("playlist", cfg) == 0
        assert "3 成功" in caplog.text, caplog.text[-300:]

    def test_partial_success_is_loud(self, tmp_path, monkeypatch, caplog):
        """**部分成功仍要吵**——批次跑時中途的 log 早被沖掉，
        收尾那一行是唯一會被讀的。"""
        cfg = _cfg(tmp_path)
        _stub_stages(monkeypatch)

        def mixed(video_id, c, w):
            if video_id == _VIDS[0]:
                return _ok_fetch(video_id, c, w)
            return _unavailable(video_id, c, w)

        monkeypatch.setattr("weft.stages.local.s0_fetch", mixed)
        with caplog.at_level("INFO"):
            rc = run_prepare("playlist", cfg)

        assert rc == 0, "有一支成功時，批次語意是繼續，不是整批失敗"
        assert "不是全部都進來了" in caplog.text
        assert "1 成功" in caplog.text and "2 不可用" in caplog.text
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert errors, "部分失敗必須是 ERROR 級，不能只是 INFO"

    def test_skiplist_is_recorded_for_each_unavailable(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        _stub_stages(monkeypatch)
        monkeypatch.setattr("weft.stages.local.s0_fetch", _unavailable)
        run_prepare("playlist", cfg)

        skips = json.loads((cfg.out_dir / "skiplist.json").read_text(encoding="utf-8"))
        assert set(skips) == set(_VIDS)
        assert all("403" in v for v in skips.values()), "原始錯誤訊息不得被吃掉"
