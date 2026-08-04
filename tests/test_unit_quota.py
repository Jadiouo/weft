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
    assert ledger.safe_limit() == int(QuotaConfig().requests_per_day * 0.9)


def test_check_passes_below_the_safe_limit(ledger: QuotaLedger):
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    ledger.check(planned_requests=1, at=moment)  # 不應 raise


def test_check_raises_before_exceeding_the_safe_limit(ledger: QuotaLedger):
    """關鍵：在**送出前**擋下，不是等 429 回來。

    §6.1：「429 會讓做到一半的 segment 白費。」
    """
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    for _ in range(ledger.safe_limit()):
        ledger.record("m", 10, 1, None, "ok", at=moment)

    with pytest.raises(QuotaExhausted) as excinfo:
        ledger.check(planned_requests=1, at=moment)
    assert "下次重置" in str(excinfo.value)


def test_check_accounts_for_the_whole_planned_batch(ledger: QuotaLedger):
    """§4.7 的批次策略會一次送 2–3 個 segment。若只檢查「還剩 ≥1 次」，
    批次的後半仍會撞牆。"""
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    for _ in range(ledger.safe_limit() - 2):
        ledger.record("m", 10, 1, None, "ok", at=moment)

    ledger.check(planned_requests=2, at=moment)  # 剛好用完，可以
    with pytest.raises(QuotaExhausted):
        ledger.check(planned_requests=3, at=moment)


def test_remaining_never_goes_negative(ledger: QuotaLedger):
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    for _ in range(ledger.safe_limit() + 50):
        ledger.record("m", 10, 1, None, "ok", at=moment)
    assert ledger.remaining(at=moment) == 0


def test_safety_margin_leaves_room_below_the_real_quota(ledger: QuotaLedger):
    """安全水位必須嚴格小於真實配額——否則「主動節流」與撞 429 沒有差別。"""
    assert ledger.safe_limit() < ledger.cfg.requests_per_day


def test_failed_requests_still_count(ledger: QuotaLedger):
    """失敗的呼叫一樣消耗額度。不計入的話節流會低估用量。"""
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    ledger.record("m", 500, 0, "v#001", "error", at=moment)
    assert ledger.usage_today(at=moment).requests == 1


# --------------------------------------------------------------------------
# 從 429 學到真實配額（SDD §9 的緩解措施）
# --------------------------------------------------------------------------


def test_observed_limit_overrides_the_configured_guess(ledger: QuotaLedger):
    """§9：「Ledger 讀取實際配額而非寫死。」

    v0.3 首跑時設定檔寫 1000（來自 SDD §6.5），實際是 20——差 50 倍，
    主動節流完全沒觸發，白燒了一整天配額。
    """
    assert ledger.limit_for("m") == ledger.cfg.requests_per_day

    ledger.record_observed_limit("m", 20)
    assert ledger.limit_for("m") == 20
    assert ledger.safe_limit("m") == 18


def test_observed_limit_is_per_model(ledger: QuotaLedger):
    """free tier 的 RPD 是 per-model 的
    （quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier）。"""
    ledger.record_observed_limit("flash-lite", 20)
    assert ledger.limit_for("flash-lite") == 20
    assert ledger.limit_for("other") == ledger.cfg.requests_per_day


def test_observed_limit_persists_across_instances(tmp_path):
    """學到的配額要跨執行保留——否則每天都得再撞一次牆才學會。"""
    cfg = QuotaConfig()
    QuotaLedger(tmp_path / "q.db", cfg).record_observed_limit("m", 20)
    assert QuotaLedger(tmp_path / "q.db", cfg).limit_for("m") == 20


def test_check_uses_the_observed_limit(ledger: QuotaLedger):
    moment = datetime(2026, 8, 4, 10, 0, tzinfo=PT)
    ledger.record_observed_limit("m", 20)
    for _ in range(18):  # 20 × 0.9
        ledger.record("m", 10, 1, None, "ok", at=moment)

    with pytest.raises(QuotaExhausted, match="實測配額"):
        ledger.check(planned_requests=1, model="m", at=moment)


def test_default_guess_is_conservative():
    """預設值寧可低估。高估的代價是白燒一整天配額（實測過了）；
    低估的代價只是提早停，隔天續跑。"""
    assert QuotaConfig().requests_per_day <= 50


# --------------------------------------------------------------------------
# 錯誤分類（v0.3 首跑的教訓）
# --------------------------------------------------------------------------


def test_permanent_errors_are_not_retried():
    """404「模型不存在」重試不會成功，只會多燒配額。

    實測：v0.3 首跑時 gemini-2.5-flash-lite 回 404（對新使用者已停用），
    15 個批次各重試 2 次＝45 次呼叫，把 20 RPD 燒光，0 個 segment 完成。
    """
    from weft.stages.understand import PermanentApiError, with_retries

    calls = []

    def always_404():
        calls.append(1)
        raise Exception("404 NOT_FOUND. {'error': {'code': 404, 'message': 'gone'}}")

    with pytest.raises(PermanentApiError):
        with_retries(always_404, max_retries=2, backoff_sec=0.01)
    assert len(calls) == 1, f"永久性錯誤被重試了 {len(calls)} 次"


def test_quota_errors_stop_immediately_and_carry_the_limit():
    """429 不重試，且要把 API 回報的真實配額帶出來。"""
    from weft.stages.understand import QuotaHit, with_retries

    calls = []

    def quota_exceeded():
        calls.append(1)
        raise Exception(
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': "
            "[{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', "
            "'quotaValue': '20'}]}}"
        )

    with pytest.raises(QuotaHit) as excinfo:
        with_retries(quota_exceeded, max_retries=2, backoff_sec=0.01)
    assert len(calls) == 1
    assert excinfo.value.limit == 20


def test_transient_errors_are_still_retried():
    """503 之類的暫時性錯誤仍應重試——不然一次網路抖動就放棄整批。"""
    from weft.stages.understand import with_retries

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise Exception("503 UNAVAILABLE. {'error': {'code': 503}}")
        return "ok"

    assert with_retries(flaky, max_retries=2, backoff_sec=0.01) == "ok"
    assert len(calls) == 3


def test_every_attempt_is_reported_for_accounting():
    """`on_attempt` 必須在**每次**呼叫後被叫到——記「批次」而不記「呼叫」
    會讓帳本嚴重低估（實測差 2.6 倍）。"""
    from weft.stages.understand import with_retries

    attempts = []

    def flaky():
        if len(attempts) < 2:
            raise Exception("503 UNAVAILABLE. {'error': {'code': 503}}")
        return "ok"

    with_retries(flaky, max_retries=3, backoff_sec=0.01,
                 on_attempt=lambda ok, exc: attempts.append(ok))
    assert attempts == [False, False, True]
