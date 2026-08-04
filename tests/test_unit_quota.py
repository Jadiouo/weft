"""額度帳本。SDD §6.1、§6.2。

§5.5 #13：不得靠撞 429 探測額度上限；必須主動用 quota ledger 預估與節流。
§6.2：RPD 於**太平洋時間**午夜重置——不是台灣午夜，也不得寫死時差。

時區是這個模組最容易寫錯、又最難察覺的地方：寫死 UTC+8 的話，夏令時間
換冬令時間那天會整整差一小時，表現是「明明該重置了卻還在擋」，而且一年
只會出錯兩次。所以這裡逐條釘住。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from weft.config import QuotaConfig
from weft.quota import QuotaExhausted, QuotaLedger

PT = ZoneInfo("America/Los_Angeles")
TAIPEI = ZoneInfo("Asia/Taipei")


@pytest.fixture
def ledger(tmp_path) -> QuotaLedger:
    return QuotaLedger(tmp_path / "quota.db", QuotaConfig())


# --------------------------------------------------------------------------
# 配額日與重置時刻（§6.2）
# --------------------------------------------------------------------------


def test_reset_timezone_is_pacific_not_taipei():
    assert QuotaConfig().reset_timezone == "America/Los_Angeles"


def test_quota_day_follows_pacific_date(ledger: QuotaLedger):
    """台北時間 2026-08-05 07:00 仍屬太平洋的 08-04。"""
    taipei_morning = datetime(2026, 8, 5, 7, 0, tzinfo=TAIPEI)
    assert ledger.quota_day(taipei_morning).isoformat() == "2026-08-04"


def test_reset_lands_at_15_taipei_during_daylight_saving(ledger: QuotaLedger):
    """§6.2：夏令時間（PDT, UTC−7）→ 台北 15:00。"""
    summer = datetime(2026, 8, 4, 12, 0, tzinfo=PT)
    reset_in_taipei = ledger.next_reset(summer).astimezone(TAIPEI)
    assert reset_in_taipei.hour == 15


def test_reset_lands_at_16_taipei_during_standard_time(ledger: QuotaLedger):
    """§6.2：冬令時間（PST, UTC−8）→ 台北 16:00。

    這條與上一條合起來就是「不得寫死時差」的意義：同一份程式碼在夏冬
    兩季必須給出不同的台北時刻。
    """
    winter = datetime(2026, 1, 15, 12, 0, tzinfo=PT)
    reset_in_taipei = ledger.next_reset(winter).astimezone(TAIPEI)
    assert reset_in_taipei.hour == 16


def test_reset_is_pacific_midnight(ledger: QuotaLedger):
    moment = datetime(2026, 8, 4, 23, 59, tzinfo=PT)
    reset = ledger.next_reset(moment)
    assert (reset.hour, reset.minute) == (0, 0)
    assert reset.date() == moment.date() + timedelta(days=1)


def test_naive_datetime_is_rejected(ledger: QuotaLedger):
    """沒有時區的時間會讓配額日算錯，且錯得很安靜。寧可炸掉。"""
    with pytest.raises(ValueError, match="時區"):
        ledger.quota_day(datetime(2026, 8, 4, 12, 0))


# --------------------------------------------------------------------------
# 記錄與統計
# --------------------------------------------------------------------------


def test_usage_accumulates_within_a_quota_day(ledger: QuotaLedger):
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    for i in range(3):
        ledger.record("gemini-2.5-flash-lite", 1000, 200, f"v#{i:03d}", "ok", at=moment)

    usage = ledger.usage_today(at=moment)
    assert usage.requests == 3
    assert usage.input_tokens == 3000
    assert usage.output_tokens == 600


def test_usage_resets_across_the_pacific_midnight(ledger: QuotaLedger):
    before = datetime(2026, 8, 4, 23, 30, tzinfo=PT)
    after = datetime(2026, 8, 5, 0, 30, tzinfo=PT)
    ledger.record("m", 100, 10, None, "ok", at=before)

    assert ledger.usage_today(at=before).requests == 1
    assert ledger.usage_today(at=after).requests == 0


def test_usage_can_be_filtered_by_model(ledger: QuotaLedger):
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    ledger.record("flash-lite", 100, 10, None, "ok", at=moment)
    ledger.record("qwen-local", 100, 10, None, "ok", at=moment)

    assert ledger.usage_today("flash-lite", at=moment).requests == 1
    assert ledger.usage_today(at=moment).requests == 2


def test_ledger_persists_across_instances(tmp_path):
    """跨日續跑的前提：重啟後帳本還在（§6.3）。"""
    cfg = QuotaConfig()
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    QuotaLedger(tmp_path / "q.db", cfg).record("m", 100, 10, None, "ok", at=moment)

    assert QuotaLedger(tmp_path / "q.db", cfg).usage_today(at=moment).requests == 1


# --------------------------------------------------------------------------
# 主動節流（§6.1、§5.5 #13）
# --------------------------------------------------------------------------


def test_safe_limit_is_ninety_percent_of_quota(ledger: QuotaLedger):
    """§6.1：安全水位設為配額的 90%。"""
    assert ledger.safe_limit == int(1000 * 0.9)


def test_check_passes_below_the_safe_limit(ledger: QuotaLedger):
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    ledger.check(planned_requests=1, at=moment)  # 不應 raise


def test_check_raises_before_exceeding_the_safe_limit(ledger: QuotaLedger):
    """關鍵：在**送出前**擋下，不是等 429 回來。

    §6.1：「429 會讓做到一半的 segment 白費。」
    """
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    for _ in range(ledger.safe_limit):
        ledger.record("m", 10, 1, None, "ok", at=moment)

    with pytest.raises(QuotaExhausted) as excinfo:
        ledger.check(planned_requests=1, at=moment)
    assert "下次重置" in str(excinfo.value)


def test_check_accounts_for_the_whole_planned_batch(ledger: QuotaLedger):
    """§4.7 的批次策略會一次送 2–3 個 segment。若只檢查「還剩 ≥1 次」，
    批次的後半仍會撞牆。"""
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    for _ in range(ledger.safe_limit - 2):
        ledger.record("m", 10, 1, None, "ok", at=moment)

    ledger.check(planned_requests=2, at=moment)  # 剛好用完，可以
    with pytest.raises(QuotaExhausted):
        ledger.check(planned_requests=3, at=moment)


def test_remaining_never_goes_negative(ledger: QuotaLedger):
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    for _ in range(ledger.safe_limit + 50):
        ledger.record("m", 10, 1, None, "ok", at=moment)
    assert ledger.remaining(at=moment) == 0


def test_safety_margin_leaves_room_below_the_real_quota(ledger: QuotaLedger):
    """安全水位必須嚴格小於真實配額——否則「主動節流」與撞 429 沒有差別。"""
    assert ledger.safe_limit < ledger.cfg.requests_per_day


def test_failed_requests_still_count(ledger: QuotaLedger):
    """失敗的呼叫一樣消耗額度。不計入的話節流會低估用量。"""
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    ledger.record("m", 500, 0, "v#001", "error", at=moment)
    assert ledger.usage_today(at=moment).requests == 1
