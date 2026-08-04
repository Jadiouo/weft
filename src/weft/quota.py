"""額度帳本。SDD §6.1、§6.2。

「**主動節流**：每次呼叫前估算 token 用量，與當日已用量相加，超過安全水位
（設為配額的 90%）即停止本日處理。**不靠撞 429**——429 會讓做到一半的
segment 白費。」（§6.1）

§5.5 #13 明文禁止靠撞 429 探測額度上限。
§6.2：RPD 於**太平洋時間**午夜重置，用 `zoneinfo` 計算，**不寫死時差**。
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    quota_day     TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    request_count INTEGER NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    segment_id    TEXT,
    status        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_by_day ON usage(quota_day, model);
"""


class QuotaExhausted(Exception):
    """本日安全水位已達。停止並記錄進度，等下次重置。

    §5.5 #6：**不得**因此靜默改用本地模型。
    """


@dataclass(frozen=True)
class Usage:
    requests: int
    input_tokens: int
    output_tokens: int


class QuotaLedger:
    """SQLite 帳本。SDD §3.1 的 `out/quota.db`。"""

    def __init__(self, path: Path, cfg) -> None:
        self.path = path
        self.cfg = cfg
        self.timezone = ZoneInfo(cfg.reset_timezone)
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    # -- 重置時刻（§6.2）------------------------------------------------

    def quota_day(self, at: datetime | None = None) -> date:
        """某個時刻屬於哪一個配額日。

        以**太平洋時間**的日期為準。§6.2 特別提醒不是台灣午夜，也不要寫死
        時差——夏令／冬令時間的差異由 zoneinfo 處理。
        """
        moment = at or datetime.now(tz=self.timezone)
        if moment.tzinfo is None:
            raise ValueError("需要帶時區的 datetime——naive datetime 會讓配額日算錯")
        return moment.astimezone(self.timezone).date()

    def next_reset(self, at: datetime | None = None) -> datetime:
        """下一次配額重置的時刻（太平洋時間午夜）。"""
        moment = (at or datetime.now(tz=self.timezone)).astimezone(self.timezone)
        tomorrow = moment.date() + timedelta(days=1)
        return datetime.combine(tomorrow, datetime.min.time(), tzinfo=self.timezone)

    # -- 記錄與查詢 ------------------------------------------------------

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        segment_id: str | None,
        status: str,
        at: datetime | None = None,
    ) -> None:
        moment = (at or datetime.now(tz=self.timezone)).astimezone(self.timezone)
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                "INSERT INTO usage (timestamp, quota_day, model, request_count,"
                " input_tokens, output_tokens, segment_id, status)"
                " VALUES (?, ?, ?, 1, ?, ?, ?, ?)",
                (
                    moment.isoformat(timespec="seconds"),
                    self.quota_day(moment).isoformat(),
                    model,
                    input_tokens,
                    output_tokens,
                    segment_id,
                    status,
                ),
            )
            conn.commit()

    def usage_today(self, model: str | None = None, at: datetime | None = None) -> Usage:
        day = self.quota_day(at).isoformat()
        query = (
            "SELECT COALESCE(SUM(request_count),0), COALESCE(SUM(input_tokens),0),"
            " COALESCE(SUM(output_tokens),0) FROM usage WHERE quota_day = ?"
        )
        params: list = [day]
        if model:
            query += " AND model = ?"
            params.append(model)
        with closing(sqlite3.connect(self.path)) as conn:
            row = conn.execute(query, params).fetchone()
        return Usage(*row)

    # -- 主動節流（§6.1）-------------------------------------------------

    @property
    def safe_limit(self) -> int:
        """安全水位：配額的 `safety_ratio`。超過即停止本日處理。"""
        return int(self.cfg.requests_per_day * self.cfg.safety_ratio)

    def remaining(self, model: str | None = None, at: datetime | None = None) -> int:
        return max(0, self.safe_limit - self.usage_today(model, at).requests)

    def check(self, planned_requests: int = 1, model: str | None = None,
              at: datetime | None = None) -> None:
        """呼叫**前**檢查。超過安全水位就 raise，不送出請求。

        這是 §5.5 #13 的落實：主動預估與節流，不靠撞 429。
        """
        used = self.usage_today(model, at).requests
        if used + planned_requests > self.safe_limit:
            raise QuotaExhausted(
                f"本日已用 {used} 次請求，再送 {planned_requests} 次會超過安全水位 "
                f"{self.safe_limit}（配額 {self.cfg.requests_per_day} × "
                f"{self.cfg.safety_ratio}）。下次重置：{self.next_reset(at).isoformat()}"
            )

    def summary(self, at: datetime | None = None) -> str:
        usage = self.usage_today(at=at)
        return (
            f"配額日 {self.quota_day(at)}：{usage.requests}/{self.safe_limit} 次請求"
            f"（input {usage.input_tokens:,} / output {usage.output_tokens:,} tokens）"
        )
