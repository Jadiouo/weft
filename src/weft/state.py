"""階段狀態與參數 hash。SDD §6.3 續跑語意。

核心規則：參數 hash 變更 → 該階段及其**下游**失效；上游不動。
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Stage(StrEnum):
    S0_FETCH = "S0"
    S1A_TRANSCRIPT = "S1a"
    S1B_SLIDES = "S1b"
    S3_ALIGN = "S3"
    S4_UNDERSTAND = "S4"
    S5_SYNTHESIZE = "S5"
    S6_RENDER = "S6"


#: 直接上游。SDD §2.2 的資料流圖。
DEPENDENCIES: dict[Stage, tuple[Stage, ...]] = {
    Stage.S0_FETCH: (),
    Stage.S1A_TRANSCRIPT: (Stage.S0_FETCH,),
    Stage.S1B_SLIDES: (Stage.S0_FETCH,),
    Stage.S3_ALIGN: (Stage.S1A_TRANSCRIPT, Stage.S1B_SLIDES),
    Stage.S4_UNDERSTAND: (Stage.S3_ALIGN,),
    Stage.S5_SYNTHESIZE: (Stage.S4_UNDERSTAND,),
    Stage.S6_RENDER: (Stage.S5_SYNTHESIZE,),
}

#: prepare 與 understand 兩個入口的階段劃分。SDD §6.4。
PREPARE_STAGES: tuple[Stage, ...] = (
    Stage.S0_FETCH,
    Stage.S1A_TRANSCRIPT,
    Stage.S1B_SLIDES,
    Stage.S3_ALIGN,
)
UNDERSTAND_STAGES: tuple[Stage, ...] = (
    Stage.S4_UNDERSTAND,
    Stage.S5_SYNTHESIZE,
    Stage.S6_RENDER,
)


def downstream_of(stage: Stage) -> set[Stage]:
    """遞移閉包：所有直接或間接依賴 `stage` 的階段。"""
    result: set[Stage] = set()
    frontier = [stage]
    while frontier:
        current = frontier.pop()
        for candidate, deps in DEPENDENCIES.items():
            if current in deps and candidate not in result:
                result.add(candidate)
                frontier.append(candidate)
    return result


def params_hash(params: dict) -> str:
    """對階段參數取穩定 hash。float 一律以 repr 序列化，避免平台差異。"""
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=repr)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class StageStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"  # 例如詞庫為空時的 S2c（§4.5 失敗行為）


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StageStatus = StageStatus.PENDING
    params_hash: str | None = None
    finished_at: str | None = None
    error: str | None = None
    note: str | None = None


class VideoState(BaseModel):
    """state.json。"""

    model_config = ConfigDict(extra="forbid")

    video_id: str
    stages: dict[Stage, StageRecord] = Field(default_factory=dict)
    #: S4 的 segment 級斷點（§6.3）。存已完成的 segment_id。
    understood_segments: list[str] = Field(default_factory=list)

    def record(self, stage: Stage) -> StageRecord:
        return self.stages.setdefault(stage, StageRecord())

    def is_satisfied(self, stage: Stage, expected_hash: str) -> bool:
        """該階段是否已完成且參數未變。"""
        rec = self.stages.get(stage)
        return (
            rec is not None
            and rec.status is StageStatus.DONE
            and rec.params_hash == expected_hash
        )

    def mark_done(self, stage: Stage, expected_hash: str, note: str | None = None) -> None:
        rec = self.record(stage)
        rec.status = StageStatus.DONE
        rec.params_hash = expected_hash
        rec.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rec.error = None
        rec.note = note

    def mark_failed(self, stage: Stage, error: str) -> None:
        rec = self.record(stage)
        rec.status = StageStatus.FAILED
        rec.error = error
        rec.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def invalidate_downstream(self, stage: Stage) -> set[Stage]:
        """參數 hash 變更時呼叫。回傳被清掉的階段集合。"""
        affected = downstream_of(stage)
        for s in affected:
            if s in self.stages:
                self.stages[s] = StageRecord()
        if Stage.S4_UNDERSTAND in affected:
            self.understood_segments = []
        return affected

    @classmethod
    def load(cls, path: Path) -> VideoState:
        """讀取 state.json，**容忍已移除的階段**。

        管線改版時（例如 v0.3 移除 S2／S2b／S2c）舊的 state.json 會留著
        已不存在的階段名。直接 validate 會拋 ValidationError，使用者的
        整個 work/ 目錄變成無法載入——那等於強迫他重跑幾十支影片的抽幀。

        未知階段**丟棄並記錄**：它們的產物本來就不再被任何階段讀取，
        留著只會讓 sync_state 拿它去比對不存在的參數 hash。
        """
        if not path.exists():
            raise FileNotFoundError(path)

        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {s.value for s in Stage}
        unknown = [k for k in (raw.get("stages") or {}) if k not in known]
        if unknown:
            for key in unknown:
                raw["stages"].pop(key)
            logging.getLogger(__name__).info(
                "%s：忽略已移除的階段 %s（管線改版後的遺留）", path.name, unknown
            )
        return cls.model_validate(raw)

    @classmethod
    def load_or_new(cls, path: Path, video_id: str) -> VideoState:
        if path.exists():
            return cls.load(path)
        return cls(video_id=video_id)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)  # 原子寫入：批次跑中途被 kill 不該留下半個 state.json
