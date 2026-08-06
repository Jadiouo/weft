"""work/ 與 out/ 的檔案佈局。SDD §3.1 是唯一權威——路徑只在這裡硬編碼一次。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkPaths:
    """單支影片在 work/ 底下的所有落地檔案。"""

    root: Path
    video_id: str

    @property
    def dir(self) -> Path:
        return self.root / self.video_id

    # S0
    @property
    def meta(self) -> Path:
        return self.dir / "00_meta.json"

    @property
    def video(self) -> Path:
        return self.dir / "01_video.mp4"

    @property
    def captions(self) -> Path:
        return self.dir / "01_captions.vtt"

    # S1b
    @property
    def frames_dir(self) -> Path:
        return self.dir / "02_frames"

    @property
    def candidates(self) -> Path:
        return self.dir / "02_candidates.json"

    @property
    def slides_dir(self) -> Path:
        return self.dir / "03_slides"

    def slide_image(self, index: int) -> Path:
        return self.slides_dir / f"slide_{index:03d}.png"

    # S1a
    @property
    def dedup(self) -> Path:
        #: S1c 去重結果（§4.3b）。用 04 這個空號——S2 於 v0.3 移除後留下的。
        return self.dir / "04_dedup.json"

    @property
    def transcript(self) -> Path:
        return self.dir / "05_transcript.json"

    # S3
    @property
    def segments(self) -> Path:
        return self.dir / "06_segments.json"

    # S4
    @property
    def understanding_dir(self) -> Path:
        return self.dir / "07_understanding"

    def understanding(self, seg_index: int) -> Path:
        return self.understanding_dir / f"seg_{seg_index:03d}.json"

    # 最終 IR
    @property
    def video_ir(self) -> Path:
        return self.dir / "08_video.json"

    @property
    def state(self) -> Path:
        return self.dir / "state.json"

    def ensure_dirs(self) -> None:
        for d in (self.dir, self.frames_dir, self.slides_dir, self.understanding_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class OutPaths:
    """out/ 的產品與帳本。"""

    root: Path

    @property
    def chunks(self) -> Path:
        return self.root / "chunks.jsonl"

    @property
    def debug_dir(self) -> Path:
        return self.root / "debug"

    def debug_md(self, video_id: str) -> Path:
        return self.debug_dir / f"{video_id}.md"

    @property
    def unverified(self) -> Path:
        return self.debug_dir / "unverified.jsonl"

    @property
    def quota_db(self) -> Path:
        return self.root / "quota.db"

    @property
    def review_log(self) -> Path:
        return self.root / "review_log.md"

    @property
    def skip_list(self) -> Path:
        return self.root / "skiplist.json"

    def ensure_dirs(self) -> None:
        for d in (self.root, self.debug_dir):
            d.mkdir(parents=True, exist_ok=True)
