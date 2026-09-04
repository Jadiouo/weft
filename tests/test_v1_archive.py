"""v1 的驗收判準：**逐字稿檔案庫**（Q5a，2026-09-04）。

v1 交付的不是 `chunks.jsonl`，是 **S0–S3 的產出**——逐字稿、時間戳、分段。
理由見 `docs/decisions.md` D35：那一層 31/31 完整、確定性、而且是 v2
要拿去重做的原料；理解層只有 8/31、CV 0.34、v2 會整個重寫。

所以驗收要驗的是**檔案庫本身有沒有變**，不是模型輸出好不好。
基線落在 `tests/golden/v1_archive_manifest.json`，是 2026-09-04 的實測值。

**沒有 `work/` 時 skip，不假通過**（CLAUDE.md 的規矩）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from weft.paths import WorkPaths
from weft.state import PREPARE_STAGES, StageStatus, VideoState

MANIFEST = Path(__file__).parent / "golden" / "v1_archive_manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]


def _cues(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else (data.get("cues") or data.get("segments") or [])


def _segments(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else (data.get("segments") or [])


def _present(cfg) -> list[str]:
    return [v for v in _manifest() if (Path(cfg.work_dir) / v / "05_transcript.json").exists()]


def test_v1_archive_integrity(cfg):
    """在場的每一支，逐字稿與分段的數字必須與基線一字不差。

    **逐支比對，不是比總和。** 總和會互相抵銷——一支掉 200 字、
    另一支多 200 字，加起來看不出來。
    """
    manifest = _manifest()
    present = _present(cfg)
    if not present:
        pytest.skip("`work/` 沒有任何 manifest 裡的影片——這台機器沒有檔案庫")

    drift = []
    for vid in present:
        want = manifest[vid]
        work = WorkPaths(cfg.work_dir, vid)
        cues = _cues(work.dir / "05_transcript.json")
        got = {
            "cues": len(cues),
            "chars": sum(
                len(c.get("text_corrected") or c.get("text_raw") or "") for c in cues
            ),
            "segments": len(_segments(work.dir / "06_segments.json")),
        }
        for key, expected in ((k, want[k]) for k in ("cues", "chars", "segments")):
            if got[key] != expected:
                drift.append(f"{vid} {key}: 基線 {expected} → 現在 {got[key]}")

    assert not drift, (
        f"逐字稿檔案庫與基線不符（在場 {len(present)}/{len(manifest)} 支）：\n"
        + "\n".join(drift)
        + "\n\n這是 v1 的交付物。數字變了代表檔案庫變了——"
        "若是刻意重跑，更新 `tests/golden/v1_archive_manifest.json` 並說明理由。"
    )


def test_v1_archive_prepare_stages_done(cfg):
    """在場的每一支，`prepare` 擁有的階段（S0–S3）全部 done。

    **只驗 S0–S3。** S4–S6 不在 v1 範圍內，而 9/1 的驗證腳本正是因為
    寫成「所有階段都要 done」，把兩支帶著 S5/S6 `pending` 的影片誤判成失敗。
    """
    present = _present(cfg)
    if not present:
        pytest.skip("`work/` 沒有任何 manifest 裡的影片")

    bad = []
    for vid in present:
        work = WorkPaths(cfg.work_dir, vid)
        state = VideoState.model_validate_json(
            (work.dir / "state.json").read_text(encoding="utf-8")
        )
        for stage in PREPARE_STAGES:
            record = state.stages.get(stage)
            if record is None or record.status is not StageStatus.DONE:
                bad.append(f"{vid} {stage}: {record.status if record else '缺'}")

    assert not bad, "S0–S3 未完成：\n" + "\n".join(bad)


def test_v1_archive_is_complete(cfg):
    """31 支全部在場。

    這條與上面兩條分開，因為它們問的是不同的問題：上面問「在場的對不對」，
    這條問「是不是全都在」。只有前者的話，`work/` 被砍到剩三支也會全綠。
    """
    manifest = _manifest()
    present = set(_present(cfg))
    if not present:
        pytest.skip("`work/` 沒有任何 manifest 裡的影片")

    missing = sorted(set(manifest) - present)
    assert not missing, (
        f"檔案庫缺 {len(missing)}/{len(manifest)} 支：{', '.join(missing)}\n"
        "重建要重新下載 + 重跑 Whisper（`.mp4` 若還在則只需後者）。"
    )


def test_manifest_records_transcript_source():
    """基線必須記下每支的逐字稿來源。

    **人工字幕與 Whisper 的可信度差一個量級**，而 v1 的 27 支是 Whisper、
    算不出 CER（沒有人工字幕可比）。混在一起看會讓「檔案庫很完整」
    掩蓋「其中 87% 沒有任何品質量測」。
    """
    manifest = _manifest()
    sources = {v["source"] for v in manifest.values()}
    assert sources <= {"whisper", "manual_caption"}, f"未知的逐字稿來源：{sources}"
    whisper = [v for v in manifest.values() if v["source"] == "whisper"]
    assert len(whisper) == 27, (
        f"Whisper 逐字稿從 27 支變成 {len(whisper)} 支。"
        "這個數字釘在這裡是因為它們是**沒有 CER 可算**的那批。"
    )
