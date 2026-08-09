"""S4c 的產出量（2026-08-09）。

**通過率上升有可能只是因為寫得比較少。**

實測：把 `depth_alpha` 從 −0.5 調到 +0.75（分段品質大幅改善，
WindowDiff 0.490 → 0.359），`2FjApOVIbUs` 的**總產出字數掉了 47%**，
而溯源通過率反而從 0.979 升到 **1.000**——分母變小，
所有既有指標都說變好了。

機制：**S4c 每段產出的 block 數與段落長短無關**（實測 1.2–1.9 個），
prompt 裡也沒有任何規定。段數砍半 → 內容砍半。

D31 是同一個形狀（prompt 改動讓 block 變少、通過率上升），
那次加的 `_MAX_BARREN_RATIO` 擋得住「整段空白」，
**擋不住「每段都少寫一半」**。
"""

from __future__ import annotations

import pytest

from weft.stages.render import content_yield
from weft.validation.thresholds import MIN_CHARS_PER_1K_SOURCE


def _ir_with(block_texts: list[list[str]], seconds: float = 60.0):
    """做一份有 N 段、每段若干 block 的 IR。用真的 model，不是 stub。"""
    from weft.ir import (
        ContentBlock, ContentType, Provenance, ProvenanceKind, Segment,
        SegmentMode, Understanding, VideoIR, VideoMeta,
    )

    segments = []
    for i, texts in enumerate(block_texts):
        segments.append(Segment(
            segment_id=f"vid#{i:03d}", video_id="vid", t_start=i * seconds,
            t_end=(i + 1) * seconds, mode=SegmentMode.SPEAKER_ONLY,
            boundary_method="topic_shift", cue_indices=[i], slide_ref=None,
            understanding=Understanding(
                summary="s",
                content_blocks=[
                    ContentBlock(
                        type=ContentType.VERNACULAR, text=t,
                        provenance=Provenance(kind=ProvenanceKind.TRANSCRIPT,
                                              ref="cue:0"))
                    for t in texts
                ],
            ),
        ))
    return VideoIR(
        meta=VideoMeta(video_id="vid", title="t",
                       duration=seconds * len(block_texts),
                       url="https://example.invalid/vid",
                       video_path="01_video.mp4"),
        segments=segments,
    )


def _transcript(total_chars: int):
    from weft.ir import Transcript, TranscriptCue, TranscriptSource

    cues = [TranscriptCue(index=0, t_start=0.0, t_end=60.0,
                          text_raw="字" * total_chars)]
    return Transcript(video_id="vid", source=TranscriptSource.WHISPER,
                      cues=cues, raw_hash=Transcript.compute_raw_hash(cues))


class TestTheMetricItself:
    def test_compression_ratio_is_independent_of_material_length(self):
        """**主指標必須與素材長短無關。**

        「字/分」是速率，短素材分母太小就沒意義——實測 90 秒的合成
        fixture 只有 19.3 字/分，而真實影片是 70–281。用了它，
        離線測試會全部誤報。
        """
        short = content_yield(_ir_with([["產出" * 25]]), _transcript(1000))
        long = content_yield(_ir_with([["產出" * 25]] * 10), _transcript(10000))
        assert short["chars_per_1k_source"] == pytest.approx(
            long["chars_per_1k_source"], rel=0.01)

    def test_halving_the_segments_halves_the_output(self):
        """**這就是 2026-08-09 踩到的那件事。**

        S4c 每段寫差不多的量，所以段數砍半、產出砍半——
        而逐字稿沒變，壓縮比因此腰斬。
        """
        tr = _transcript(4000)
        many = content_yield(_ir_with([["內容" * 30]] * 8), tr)
        few = content_yield(_ir_with([["內容" * 30]] * 4), tr)
        assert few["chars_per_1k_source"] == pytest.approx(
            many["chars_per_1k_source"] / 2, rel=0.01)

    def test_blocks_per_segment_is_recorded(self):
        """這一欄是**診斷**：它說明產出量掉了是因為段少了還是因為每段寫少了。"""
        y = content_yield(_ir_with([["a", "b"], ["c"]]), _transcript(100))
        assert y["segments"] == 2
        assert y["blocks"] == 3
        assert y["blocks_per_segment"] == pytest.approx(1.5)

    def test_no_transcript_still_returns_the_rate_fields(self):
        """讀不到逐字稿時只記速率、不算壓縮比——新增的觀測值不該把路徑弄壞。"""
        y = content_yield(_ir_with([["abc"]]))
        assert "chars_per_min" in y
        assert "chars_per_1k_source" not in y


class TestTheFloorActuallyGates:
    """`MIN_CHARS_PER_1K_SOURCE` 不是常數斷言——它要真的擋得住東西。"""

    def test_a_collapsed_run_is_below_the_floor(self):
        """S4c 幾乎沒回東西：8000 字逐字稿只產出 30 字。"""
        y = content_yield(_ir_with([["字" * 10]] * 3), _transcript(8000))
        assert y["chars_per_1k_source"] < MIN_CHARS_PER_1K_SOURCE

    def test_the_worst_real_video_is_still_above_the_floor(self):
        """實測最低是 `2FjApOVIbUs` 的 362（調 α 之後）。

        下限若訂在會誤殺真實素材的位置，它就會被調低，然後失去意義。
        """
        y = content_yield(_ir_with([["字" * 362]]), _transcript(1000))
        assert y["chars_per_1k_source"] > MIN_CHARS_PER_1K_SOURCE

    def test_the_floor_leaves_real_headroom(self):
        """下限刻意寬鬆——它抓崩潰，不抓退步。

        訂太接近實測值的話，一次正常的素材變異就會紅，
        然後大家會習慣性地調低它（§5.5 #7 禁止的正是這件事）。
        """
        assert MIN_CHARS_PER_1K_SOURCE <= 362 / 3


class TestItIsWiredIntoS6:
    def test_s6_raises_when_output_collapses(self, tmp_path):
        """光有下限沒有用，要證明 S6 真的會擋。

        **不 mock `check_video`**——替身要猜它的介面，猜錯就變成測替身
        （§5.5 #10）。這裡的 IR 是合法的，溯源會真的跑並全數通過
        （block 的字都在逐字稿裡），所以擋下來的一定是產出量那一關。
        """
        from weft.config import Config
        from weft.paths import OutPaths, WorkPaths
        from weft.stages.cloud import s6_render

        cfg = Config()
        work = WorkPaths(tmp_path / "work", "vid")
        out = OutPaths(tmp_path / "out")
        work.ensure_dirs()
        work.transcript.write_text(_transcript(8000).model_dump_json(),
                                   encoding="utf-8")
        ir = _ir_with([["字" * 10]] * 3)

        with pytest.raises(RuntimeError, match="產出量過低"):
            s6_render(cfg, ir, work, out)
