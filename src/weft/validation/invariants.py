"""SDD §5.3 機械式不變量。

「每次跑完必須全數通過，任一失敗即中止並報錯。」

§5.5 #9：**不得把這些 assert 改成 warning 或 log。** 因此本模組刻意
不提供 `warn_only` 之類的參數——唯一的公開執行入口 `assert_all` 會 raise。
`check_all` 回傳清單是為了讓報錯訊息能一次列出所有問題，不是為了讓呼叫端
忽略它。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..ir import Chunk, ProvenanceKind, SegmentMode, Transcript, VideoIR
from .thresholds import COVERAGE_TOLERANCE_SEC


@dataclass(frozen=True)
class Violation:
    rule: int
    name: str
    detail: str

    def __str__(self) -> str:
        return f"[不變量 {self.rule}｜{self.name}] {self.detail}"


class InvariantViolation(Exception):
    """任一不變量失敗即中止。"""

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        body = "\n".join(f"  - {v}" for v in violations)
        super().__init__(f"違反 {len(violations)} 條 SDD §5.3 不變量：\n{body}")


# --------------------------------------------------------------------------
# 個別規則
# --------------------------------------------------------------------------


def rule_01_segments_disjoint(ir: VideoIR) -> list[Violation]:
    """1. segments 時間區間互不重疊。"""
    out: list[Violation] = []
    ordered = sorted(ir.segments, key=lambda s: s.t_start)
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.t_start < prev.t_end - 1e-6:
            out.append(
                Violation(
                    1,
                    "segments 互不重疊",
                    f"{prev.segment_id}[{prev.t_start:.2f}, {prev.t_end:.2f}] 與 "
                    f"{cur.segment_id}[{cur.t_start:.2f}, {cur.t_end:.2f}] 重疊",
                )
            )
    return out


def rule_02_segments_cover_video(ir: VideoIR) -> list[Violation]:
    """2. segments 的聯集等於影片全長（容忍 ±1 秒）。"""
    out: list[Violation] = []
    duration = ir.meta.duration
    if not ir.segments:
        return [Violation(2, "segments 覆蓋全片", f"segments 為空，但影片長度為 {duration:.2f}s")]

    ordered = sorted(ir.segments, key=lambda s: s.t_start)
    if ordered[0].t_start > COVERAGE_TOLERANCE_SEC:
        out.append(
            Violation(2, "segments 覆蓋全片", f"開頭有 {ordered[0].t_start:.2f}s 未被任何 segment 覆蓋")
        )
    if duration - ordered[-1].t_end > COVERAGE_TOLERANCE_SEC:
        out.append(
            Violation(
                2,
                "segments 覆蓋全片",
                f"結尾有 {duration - ordered[-1].t_end:.2f}s 未被覆蓋"
                f"（最後 segment 止於 {ordered[-1].t_end:.2f}s，全長 {duration:.2f}s）",
            )
        )
    for prev, cur in zip(ordered, ordered[1:]):
        gap = cur.t_start - prev.t_end
        if gap > COVERAGE_TOLERANCE_SEC:
            out.append(
                Violation(
                    2,
                    "segments 覆蓋全片",
                    f"{prev.segment_id} 與 {cur.segment_id} 之間有 {gap:.2f}s 空隙",
                )
            )
    return out


def rule_03_cues_assigned_once(ir: VideoIR, transcript: Transcript) -> list[Violation]:
    """3. 每一句逐字稿恰好被指派到一個 segment。"""
    out: list[Violation] = []
    assigned: dict[int, list[str]] = {}
    for seg in ir.segments:
        for idx in seg.cue_indices:
            assigned.setdefault(idx, []).append(seg.segment_id)

    all_indices = {c.index for c in transcript.cues}
    for idx in sorted(all_indices - assigned.keys()):
        out.append(Violation(3, "逐字稿指派", f"cue #{idx} 未被指派到任何 segment"))
    for idx in sorted(assigned.keys() - all_indices):
        out.append(Violation(3, "逐字稿指派", f"segment 引用了不存在的 cue #{idx}"))
    for idx, owners in sorted(assigned.items()):
        if len(owners) > 1:
            out.append(
                Violation(3, "逐字稿指派", f"cue #{idx} 被指派到 {len(owners)} 個 segment：{owners}")
            )
    return out


def rule_04_slide_refs_exist(ir: VideoIR) -> list[Violation]:
    """4. 所有 slide_ref 指向存在的 slide 物件。"""
    out: list[Violation] = []
    known = {s.slide_id for s in ir.slides}
    for seg in ir.segments:
        if seg.slide_ref is not None and seg.slide_ref not in known:
            out.append(
                Violation(4, "slide_ref 可解析", f"{seg.segment_id} 指向不存在的 slide {seg.slide_ref!r}")
            )
        if seg.mode is SegmentMode.SLIDE and seg.slide_ref is None:
            out.append(
                Violation(4, "slide_ref 可解析", f"{seg.segment_id} 的 mode 為 slide 但 slide_ref 為 null")
            )
        # v0.3：candidate_ref 是 S1b 取出的候選幀，與 VLM 的判定無關。
        # 它若指向不存在的 slide，代表 S1b/S3 的產物對不上。
        if seg.candidate_ref is not None and seg.candidate_ref not in known:
            out.append(
                Violation(4, "slide_ref 可解析",
                          f"{seg.segment_id} 的 candidate_ref 指向不存在的 slide "
                          f"{seg.candidate_ref!r}")
            )
        # VLM 判定不是投影片時，slide_ref 必須已清空——留著會讓下游以為
        # 該段有投影片來源，chunk 的 provenance 就錯了
        if (
            seg.understanding is not None
            and not seg.understanding.is_slide
            and seg.slide_ref is not None
        ):
            out.append(
                Violation(4, "slide_ref 可解析",
                          f"{seg.segment_id} 被判定不是投影片，但 slide_ref 未清空")
            )
    for seg in ir.segments:
        if seg.understanding is None:
            continue
        for i, block in enumerate(seg.understanding.content_blocks):
            if block.provenance.kind is ProvenanceKind.SLIDE_OCR and block.provenance.ref not in known:
                out.append(
                    Violation(
                        4,
                        "slide_ref 可解析",
                        f"{seg.segment_id} block#{i} 的 provenance 指向不存在的 slide "
                        f"{block.provenance.ref!r}",
                    )
                )
    return out


def rule_05_images_exist(ir: VideoIR, base_dir: Path) -> list[Violation]:
    """5. 所有 image_path 對應的檔案實際存在且可開啟。"""
    out: list[Violation] = []
    for slide in ir.slides:
        path = base_dir / slide.image_path
        if not path.exists():
            out.append(Violation(5, "投影片圖存在", f"{slide.slide_id}：{path} 不存在"))
            continue
        try:
            from PIL import Image

            with Image.open(path) as img:
                img.verify()
        except Exception as exc:  # noqa: BLE001 —— 任何開不起來的理由都算失敗
            out.append(Violation(5, "投影片圖存在", f"{slide.slide_id}：{path} 無法開啟（{exc}）"))
    return out


def rule_06_timestamps_sane(ir: VideoIR, transcript: Transcript | None = None) -> list[Violation]:
    """6. 所有時間戳單調遞增且落在 [0, duration]。"""
    out: list[Violation] = []
    duration = ir.meta.duration

    def check_range(label: str, t: float) -> None:
        if t < 0 or t > duration + COVERAGE_TOLERANCE_SEC:
            out.append(Violation(6, "時間戳範圍", f"{label} = {t:.2f}s 超出 [0, {duration:.2f}]"))

    for seg in ir.segments:
        check_range(f"{seg.segment_id}.t_start", seg.t_start)
        check_range(f"{seg.segment_id}.t_end", seg.t_end)
        if seg.t_end <= seg.t_start:
            out.append(
                Violation(6, "時間戳單調", f"{seg.segment_id} 的 t_end({seg.t_end:.2f}) ≤ t_start({seg.t_start:.2f})")
            )
    ordered = sorted(ir.segments, key=lambda s: s.t_start)
    if [s.segment_id for s in ordered] != [s.segment_id for s in ir.segments]:
        out.append(Violation(6, "時間戳單調", "segments 未依 t_start 遞增排列"))

    for slide in ir.slides:
        check_range(f"{slide.slide_id}.t_first_seen", slide.t_first_seen)
        check_range(f"{slide.slide_id}.t_last_seen", slide.t_last_seen)
        if slide.t_last_seen < slide.t_first_seen:
            out.append(
                Violation(6, "時間戳單調", f"{slide.slide_id} 的 t_last_seen < t_first_seen")
            )
        for bf in slide.build_frames:
            check_range(f"{slide.slide_id}.build_frame", bf)

    if transcript is not None:
        for prev, cur in zip(transcript.cues, transcript.cues[1:]):
            if cur.t_start < prev.t_start - 1e-6:
                out.append(
                    Violation(6, "時間戳單調", f"逐字稿 cue #{cur.index} 的 t_start 早於前一句")
                )
        for cue in transcript.cues:
            check_range(f"cue#{cue.index}.t_start", cue.t_start)
            check_range(f"cue#{cue.index}.t_end", cue.t_end)
    return out


def rule_07_provenance_not_null(ir: VideoIR) -> list[Violation]:
    """7. 每個 content_block 的 provenance 非 null。

    pydantic 已在建構時擋下缺欄的情形，這裡補上「有欄位但 ref 為空字串」
    這種形式上合法、實質上無法溯源的漏洞。
    """
    out: list[Violation] = []
    for seg in ir.segments:
        if seg.understanding is None:
            continue
        for i, block in enumerate(seg.understanding.content_blocks):
            if not block.provenance.ref.strip():
                out.append(
                    Violation(7, "provenance 非 null", f"{seg.segment_id} block#{i} 的 provenance.ref 為空")
                )
    return out


#: §5.3 #8 要求 metadata「無缺欄、無 null」。以下兩組欄位是有意的例外，
#: 理由寫在此處而非散落在程式碼裡：
#:   - series_id / episode_index：SDD §7.5 的 v2 預留欄位，單支影片（非
#:     playlist）來源時本就無值。
#:   - slide_ref：speaker_only 段落沒有投影片。改以條件檢查涵蓋——
#:     provenance_kind 為 slide_ocr 時 slide_ref 必須有值。
_CHUNK_NULLABLE_FIELDS = frozenset({"series_id", "episode_index", "slide_ref"})


def rule_08_chunk_metadata_complete(chunks: list[Chunk]) -> list[Violation]:
    """8. 每個輸出 chunk 的 metadata 欄位完整（無缺欄、無 null）。"""
    out: list[Violation] = []
    expected = set(type(chunks[0]).model_fields["metadata"].annotation.model_fields) if chunks else set()

    for chunk in chunks:
        dumped = chunk.metadata.model_dump()
        missing = expected - dumped.keys()
        if missing:
            out.append(Violation(8, "chunk metadata 完整", f"{chunk.id} 缺少欄位 {sorted(missing)}"))
        for key, value in dumped.items():
            if value is None and key not in _CHUNK_NULLABLE_FIELDS:
                out.append(Violation(8, "chunk metadata 完整", f"{chunk.id} 的 {key} 為 null"))
            if isinstance(value, str) and not value.strip():
                out.append(Violation(8, "chunk metadata 完整", f"{chunk.id} 的 {key} 為空字串"))
        if chunk.metadata.provenance_kind is ProvenanceKind.SLIDE_OCR and not chunk.metadata.slide_ref:
            out.append(
                Violation(
                    8,
                    "chunk metadata 完整",
                    f"{chunk.id} 的 provenance_kind 為 slide_ocr 但 slide_ref 為空",
                )
            )
        if not chunk.text.strip():
            out.append(Violation(8, "chunk metadata 完整", f"{chunk.id} 的 text 為空"))
        # `content_sha` 存在但**沒人核對**的話，它就只是一個看起來很可靠
        # 的字串——而下游正是要靠它判斷「id 沒變但內容變了」。
        # §5.2 的十項門檻裡有 7 項只 assert 常數值，就是這樣爛掉的。
        import hashlib

        actual = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16]
        if chunk.metadata.content_sha != actual:
            out.append(
                Violation(
                    8,
                    "chunk metadata 完整",
                    f"{chunk.id} 的 content_sha 與 text 不符"
                    f"（記的是 {chunk.metadata.content_sha}，實際 {actual}）",
                )
            )
    return out


def rule_09_transcript_raw_intact(transcript: Transcript) -> list[Violation]:
    """9. transcript_raw 在任何階段後未被修改（hash 比對）。"""
    if transcript.raw_is_intact():
        return []
    return [
        Violation(
            9,
            "transcript_raw 未被修改",
            f"raw_hash 不符：記錄為 {transcript.raw_hash[:12]}…，"
            f"實算為 {Transcript.compute_raw_hash(transcript.cues)[:12]}…",
        )
    ]


def rule_10_corrections_traceable(ir: VideoIR, transcript: Transcript) -> list[Violation]:
    """10. corrections 中每一筆的 from 字串實際出現在 transcript_raw 中。"""
    out: list[Violation] = []
    for cue in transcript.cues:
        for corr in cue.corrections:
            if corr.from_text not in cue.text_raw:
                out.append(
                    Violation(
                        10,
                        "corrections 可追溯",
                        f"cue #{cue.index} 的修正 {corr.from_text!r}→{corr.to_text!r} "
                        f"中，{corr.from_text!r} 未出現於 text_raw",
                    )
                )
    for seg in ir.segments:
        for corr in seg.corrections:
            if corr.from_text not in seg.transcript_raw:
                out.append(
                    Violation(
                        10,
                        "corrections 可追溯",
                        f"{seg.segment_id} 的修正 {corr.from_text!r}→{corr.to_text!r} "
                        f"中，{corr.from_text!r} 未出現於 transcript_raw",
                    )
                )
    return out


# --------------------------------------------------------------------------
# 執行入口
# --------------------------------------------------------------------------


def check_all(
    ir: VideoIR,
    transcript: Transcript,
    base_dir: Path,
    chunks: list[Chunk] | None = None,
) -> list[Violation]:
    """跑完全部 10 條，回傳所有違反項（不 raise）。

    回傳清單是為了讓錯誤訊息能一次列全，**不是**讓呼叫端把它降級成
    warning——那是 §5.5 #9 明文禁止的。正式流程請用 `assert_all`。
    """
    violations: list[Violation] = []
    violations += rule_01_segments_disjoint(ir)
    violations += rule_02_segments_cover_video(ir)
    violations += rule_03_cues_assigned_once(ir, transcript)
    violations += rule_04_slide_refs_exist(ir)
    violations += rule_05_images_exist(ir, base_dir)
    violations += rule_06_timestamps_sane(ir, transcript)
    violations += rule_07_provenance_not_null(ir)
    violations += rule_09_transcript_raw_intact(transcript)
    violations += rule_10_corrections_traceable(ir, transcript)
    if chunks is not None:
        violations += rule_08_chunk_metadata_complete(chunks)
    return violations


def assert_all(
    ir: VideoIR,
    transcript: Transcript,
    base_dir: Path,
    chunks: list[Chunk] | None = None,
) -> None:
    """任一失敗即中止並報錯。SDD §5.3。"""
    violations = check_all(ir, transcript, base_dir, chunks)
    if violations:
        raise InvariantViolation(violations)
