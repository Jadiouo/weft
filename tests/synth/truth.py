"""合成影片的 ground truth 模型。SDD §5.1（A）。

設計要點：ground truth **只由場景定義推導**，不手寫第二份。邊界時間、逐秒
幀分類、期望投影片張數全部是同一組 `LogicalPage` 的函數。手寫兩份遲早會
不一致，而不一致的 ground truth 會讓測試失去意義。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PageKind = Literal["slide", "speaker"]


@dataclass(frozen=True)
class LogicalPage:
    """一個「邏輯頁面」——人眼會認為是同一頁的一段時間。

    逐條動畫（A2）的六次疊加是**一個** LogicalPage，這正是 expected_slides
    要等於 1 的原因。
    """

    label: str
    kind: PageKind
    duration: float
    #: 此邏輯頁面期望被偵測成幾張投影片。speaker 頁為 0，逐條動畫頁為 1。
    expected_slides: int = 1
    #: 逐條動畫的各個 build 相對於本頁起點的時間（秒）。
    build_offsets: tuple[float, ...] = ()
    #: 期望被選為代表幀的時間窗（相對本頁起點）。逐條動畫時為最後一個 build 之後。
    keyframe_offset_window: tuple[float, float] | None = None
    #: 渲染參數，交給 renderer 解讀（版型、疊加特效等）
    render: dict = field(default_factory=dict)
    #: 這一頁講者說的話。用來合成字幕軌，讓 S1a→S3 能在合成素材上端到端跑。
    #: 每個元素為 `(文字, 該文字中被注入的 ASR 錯誤 → 正解)`，後者可為 None。
    speech: tuple[tuple[str, tuple[str, str] | None], ...] = ()


@dataclass(frozen=True)
class PlacedPage:
    """LogicalPage 加上絕對時間。"""

    index: int
    page: LogicalPage
    t_start: float

    @property
    def t_end(self) -> float:
        return self.t_start + self.page.duration

    @property
    def build_times(self) -> list[float]:
        return [self.t_start + off for off in self.page.build_offsets]

    @property
    def keyframe_window(self) -> tuple[float, float] | None:
        w = self.page.keyframe_offset_window
        return None if w is None else (self.t_start + w[0], self.t_start + w[1])

    def cues(self) -> list[tuple[float, float, str, tuple[str, str] | None]]:
        """把 speech 平均鋪在本頁的時間區間上。"""
        lines = self.page.speech
        if not lines:
            return []
        span = self.page.duration / len(lines)
        return [
            (self.t_start + i * span, self.t_start + (i + 1) * span - 0.1, text, error)
            for i, (text, error) in enumerate(lines)
        ]


@dataclass(frozen=True)
class SynthTruth:
    """一個合成場景的完整 ground truth。"""

    name: str
    description: str
    expectation: str  # SDD §5.1 表格中的「期望行為」原文
    pages: tuple[LogicalPage, ...]
    fps: int = 5
    #: 720p：S1b 偵測階段本來就會降到短邊 180px，但 03_slides/*.png 要送去
    #: OCR 與 VLM，解析度太低會讓 S2/S4 的測試結果對真實素材沒有參考價值。
    width: int = 1280
    height: int = 720

    @property
    def placed(self) -> list[PlacedPage]:
        out: list[PlacedPage] = []
        t = 0.0
        for i, page in enumerate(self.pages):
            out.append(PlacedPage(index=i, page=page, t_start=t))
            t += page.duration
        return out

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.pages)

    @property
    def slide_boundaries(self) -> list[float]:
        """換頁偵測要比對的 ground truth 邊界。

        定義：**投影片段落的起訖時刻**（去掉 0 與影片全長）。slide→speaker 與
        speaker→slide 的切換同樣是邊界，因為那也是投影片段落的起訖。相同內容
        的頁面重複出現（A7 回放）各自計為獨立段落，不因內容相同而合併。
        """
        marks: set[float] = set()
        for placed in self.placed:
            if placed.page.kind != "slide":
                continue
            marks.add(round(placed.t_start, 3))
            marks.add(round(placed.t_end, 3))
        marks.discard(0.0)
        marks.discard(round(self.duration, 3))
        return sorted(marks)

    @property
    def expected_slide_count(self) -> int:
        return sum(p.expected_slides for p in self.pages)

    @property
    def expected_merge_counts(self) -> list[int]:
        """每個 slide 邏輯頁面期望被偵測成幾張。供 §5.2 的逐條動畫合併正確率使用。"""
        return [p.expected_slides for p in self.pages if p.kind == "slide"]

    def frame_class_at(self, t: float) -> PageKind:
        for placed in self.placed:
            if placed.t_start <= t < placed.t_end:
                return placed.page.kind
        return self.placed[-1].page.kind

    def frame_classes(self, sample_fps: float = 1.0) -> list[str]:
        """逐個抽樣點的分類 ground truth。抽樣點取每格中心，避免正好落在
        切換瞬間造成標註本身有歧義。"""
        step = 1.0 / sample_fps
        n = int(self.duration / step)
        return [self.frame_class_at((i + 0.5) * step) for i in range(n)]

    @property
    def all_cues(self) -> list[tuple[float, float, str, tuple[str, str] | None]]:
        return [c for p in self.placed for c in p.cues()]

    @property
    def expected_corrections(self) -> list[tuple[int, str, str]]:
        """`(cue_index, 錯字, 正解)`。S2c 的 ground truth（§5.2 precision）。"""
        return [
            (i, err[0], err[1])
            for i, (_, _, _, err) in enumerate(self.all_cues)
            if err is not None
        ]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "expectation": self.expectation,
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "slide_boundaries": self.slide_boundaries,
            "expected_slide_count": self.expected_slide_count,
            "expected_merge_counts": self.expected_merge_counts,
            "pages": [
                {
                    "index": p.index,
                    "label": p.page.label,
                    "kind": p.page.kind,
                    "t_start": round(p.t_start, 3),
                    "t_end": round(p.t_end, 3),
                    "expected_slides": p.page.expected_slides,
                    "build_times": [round(b, 3) for b in p.build_times],
                    "keyframe_window": (
                        [round(x, 3) for x in p.keyframe_window] if p.keyframe_window else None
                    ),
                }
                for p in self.placed
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
