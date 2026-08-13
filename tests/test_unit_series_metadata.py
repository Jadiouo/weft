"""系列資訊從 playlist 流到 chunk（票 15，2026-08-13）。

票 15 原本寫的是「`series_id` 沒有流到 chunk，管線斷了」。
**查完發現前提是錯的**——`resolve_targets` → `run_prepare` → `VideoMeta`
→ `ChunkMetadata` 這條鏈是通的，八支素材是 None 只因為當初
都用單支 video_id 抓的，那時 None 是**正確**的值。

但查的過程找到一個真的缺陷：**回填不會發生**。

```python
if satisfied(Stage.S0_FETCH) and work.meta.exists():
    meta = VideoMeta.model_validate_json(...)   # 直接讀舊的
else:
    meta = local.s0_fetch(...)
    if series_id or episode_index:              # 只有這條路會蓋
```

一支影片先以單支 id 抓過、之後再用 playlist 跑時，S0 已滿足，
那段程式碼不執行，`series_id` 永遠補不上。而重抓影片沒有理由
（檔案就在那），所以這不是快取失效問題，是**旁路資訊沒有回填**。
"""

from __future__ import annotations

import json

import pytest

from weft.config import Config
from weft.ir import CandidateSet, Transcript, TranscriptCue, TranscriptSource, VideoMeta
from weft.paths import WorkPaths
from weft.pipeline import prepare_one

_VID = "aaaaaaaaaaa"


def _cfg(tmp_path):
    cfg = Config()
    cfg.work_dir = tmp_path / "work"
    cfg.out_dir = tmp_path / "out"
    cfg.survey_each_video = False
    return cfg


def _stub_all(monkeypatch):
    """S0–S3 全部換成不碰外部資源的假實作（與 `test_unit_pipeline.py` 同一套）。"""
    meta = VideoMeta(video_id=_VID, title="t", duration=90.0,
                     url=f"https://example.invalid/{_VID}", video_path="01_video.mp4")

    def fake_fetch(video_id, _cfg, work):
        work.meta.write_text(meta.model_dump_json(), encoding="utf-8")
        return meta.model_copy()

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


def _meta(cfg) -> dict:
    return json.loads(
        WorkPaths(cfg.work_dir, _VID).meta.read_text(encoding="utf-8"))


class TestTheChainWorks:
    def test_playlist_source_stamps_the_series_fields(self, tmp_path, monkeypatch):
        cfg = _cfg(tmp_path)
        _stub_all(monkeypatch)
        prepare_one(_VID, cfg, "PL123", 7)
        assert _meta(cfg)["series_id"] == "PL123"
        assert _meta(cfg)["episode_index"] == 7

    def test_single_video_source_leaves_them_null(self, tmp_path, monkeypatch):
        """**單支來源時 None 是正確的**，不是缺失。

        沒有這一條，「補上系列資訊」很容易被實作成從標題猜——
        而「古典醫學之人體設計系列-27胰腺」裡的 27 是不是集數，
        換一個系列就不一定了（v0.2／v0.4 兩次栽在這類推廣）。
        """
        cfg = _cfg(tmp_path)
        _stub_all(monkeypatch)
        prepare_one(_VID, cfg)
        assert _meta(cfg)["series_id"] is None
        assert _meta(cfg)["episode_index"] is None


class TestBackfill:
    """**這是票 15 真正的缺陷。**"""

    def test_series_info_is_backfilled_on_a_cached_fetch(self, tmp_path, monkeypatch):
        """先以單支 id 抓過、之後用 playlist 跑 → 要補上。

        S0 這時是「已滿足」的，舊程式碼因此整段跳過。
        """
        cfg = _cfg(tmp_path)
        _stub_all(monkeypatch)

        prepare_one(_VID, cfg)
        assert _meta(cfg)["series_id"] is None

        # 第二次帶著 playlist 資訊來——S0 不會重抓，但資訊要補上
        prepare_one(_VID, cfg, "PL123", 3)
        assert _meta(cfg)["series_id"] == "PL123", "S0 已滿足時系列資訊沒有回填"
        assert _meta(cfg)["episode_index"] == 3

    def test_backfill_does_not_refetch(self, tmp_path, monkeypatch):
        """**只補 metadata，不重抓影片。** 重抓沒有理由——檔案就在那。"""
        cfg = _cfg(tmp_path)
        _stub_all(monkeypatch)
        prepare_one(_VID, cfg)

        calls = []
        real = __import__("weft.stages.local", fromlist=["s0_fetch"]).s0_fetch

        def counting(video_id, c, work):
            calls.append(video_id)
            return real(video_id, c, work)

        monkeypatch.setattr("weft.stages.local.s0_fetch", counting)
        prepare_one(_VID, cfg, "PL123", 3)
        assert calls == [], "回填不該觸發重抓"
        assert _meta(cfg)["series_id"] == "PL123"

    def test_no_rewrite_when_nothing_changed(self, tmp_path, monkeypatch):
        """值沒變就不要動檔案——無謂的寫入會讓 mtime 變，混淆事後追查。"""
        cfg = _cfg(tmp_path)
        _stub_all(monkeypatch)
        prepare_one(_VID, cfg, "PL123", 3)
        path = WorkPaths(cfg.work_dir, _VID).meta
        before = path.stat().st_mtime_ns

        prepare_one(_VID, cfg, "PL123", 3)
        assert path.stat().st_mtime_ns == before


class TestItReachesTheChunk:
    def test_chunk_metadata_carries_the_series_fields(self):
        """鏈的最後一段：`VideoMeta` → `ChunkMetadata`。

        中間斷掉的話，上面那些測試全綠而 vault 仍然拿不到系列資訊。
        """
        from weft.ir import (
            ContentBlock, ContentType, Provenance, ProvenanceKind, Segment,
            SegmentMode, Understanding, VideoIR,
        )
        from weft.stages.render import build_chunks

        meta = VideoMeta(video_id=_VID, title="t", duration=90.0,
                         url=f"https://example.invalid/{_VID}",
                         video_path="01_video.mp4",
                         series_id="PL123", episode_index=7)
        seg = Segment(
            segment_id=f"{_VID}#000", video_id=_VID, t_start=0.0, t_end=90.0,
            mode=SegmentMode.SPEAKER_ONLY, boundary_method="topic_shift",
            cue_indices=[0], slide_ref=None,
            understanding=Understanding(
                summary="s",
                content_blocks=[ContentBlock(
                    type=ContentType.VERNACULAR, text="內容",
                    provenance=Provenance(kind=ProvenanceKind.TRANSCRIPT,
                                          ref="cue:0"))],
            ),
        )
        chunks, _ = build_chunks(VideoIR(meta=meta, segments=[seg]), Config().s6)
        assert chunks
        assert chunks[0].metadata.series_id == "PL123"
        assert chunks[0].metadata.episode_index == 7
