"""取樣溫度必須進 S4c 的冪等鍵（2026-08-13）。

**這是同一類 bug 的第六次**，所以這次在做出來的當下就留測試：

| | 漏掉的東西 |
|---|---|
| D20 / D22 | 早期的階段參數 |
| D30 | `prompt_version`（未錨定的字串替換 bump 錯了類別）|
| D32 | `segment_id` 的位置性——S4c 讀到別的時間範圍的快取 |
| α（2026-08-09）| `DEPTH_ALPHA` 是模組常數，不在 `cfg.s3` 裡 |
| **溫度** | 原本寫死在 `providers.generate()` 的預設值 |

**冪等鍵要涵蓋所有決定結果的東西。** 溫度直接決定取樣，
而 R44 實測它造成 2.31 倍的產出量變異——它不在鍵裡的話，
改了溫度會靜靜命中舊快取，量出「改了沒效果」。
"""

from __future__ import annotations

import pytest

from weft.config import Config


def _understanding(temperature, model="ollama:qwen2.5:14b", version="v7"):
    from weft.ir import Understanding

    return Understanding(
        summary="s", model_used=model, prompt_version=version,
        input_fingerprint="fp", temperature=temperature,
    )


def _segment():
    from weft.ir import Segment, SegmentMode

    return Segment(
        segment_id="vid#000", video_id="vid", t_start=0.0, t_end=60.0,
        mode=SegmentMode.SPEAKER_ONLY, boundary_method="topic_shift",
        transcript_raw="內容", slide_ref=None, cue_indices=[0],
    )


class TestItIsInTheKey:
    def test_config_has_the_field(self):
        assert hasattr(Config().s4, "temperature")

    def test_changing_it_changes_the_stage_params(self):
        """階段層的參數指紋——`sync_state` 靠它作廢下游。"""
        from weft.pipeline import stage_params
        from weft.state import Stage

        a, b = Config(), Config()
        b.s4.temperature = a.s4.temperature + 0.5
        assert (stage_params(a, Stage.S4_UNDERSTAND)
                != stage_params(b, Stage.S4_UNDERSTAND))

    def test_cached_result_at_a_different_temperature_is_rejected(self, tmp_path,
                                                                  monkeypatch):
        """**逐段快取才是真正會咬人的那一層。**

        S4c 不是用階段參數指紋決定要不要重跑，是逐段比對
        `model + prompt_version + input_fingerprint`。溫度沒進去的話，
        改了溫度會靜靜命中舊快取。
        """
        from weft.paths import WorkPaths
        from weft.stages import cloud

        cfg = Config().s4
        cfg.model = "ollama:qwen2.5:14b"
        cfg.temperature = 0.0
        work = WorkPaths(tmp_path / "work", "vid")
        work.ensure_dirs()
        seg = _segment()

        monkeypatch.setattr(cloud, "_index_of", lambda *_a: 0)
        monkeypatch.setattr(cloud, "segment_fingerprint", lambda *_a: "fp")

        # 溫度不同的快取 → 不得命中
        work.understanding(0).write_text(
            _understanding(0.2).model_dump_json(), encoding="utf-8")
        assert cloud._load_cached(work, seg, cfg) is None

        # 溫度相同 → 命中
        work.understanding(0).write_text(
            _understanding(0.0).model_dump_json(), encoding="utf-8")
        assert cloud._load_cached(work, seg, cfg) is not None

    def test_cache_without_the_field_is_rejected(self, tmp_path, monkeypatch):
        """舊快取沒有這個欄位（None）→ **保守重跑**。

        讓它命中等於相信一個無法驗證的假設：「那份結果大概是 0.2 產的」。
        D32 的快取就是這樣讀到別的時間範圍的內容。
        """
        from weft.paths import WorkPaths
        from weft.stages import cloud

        cfg = Config().s4
        cfg.model = "ollama:qwen2.5:14b"
        cfg.temperature = 0.2
        work = WorkPaths(tmp_path / "work", "vid")
        work.ensure_dirs()
        monkeypatch.setattr(cloud, "_index_of", lambda *_a: 0)
        monkeypatch.setattr(cloud, "segment_fingerprint", lambda *_a: "fp")

        work.understanding(0).write_text(
            _understanding(None).model_dump_json(), encoding="utf-8")
        assert cloud._load_cached(work, _segment(), cfg) is None


class TestItReachesTheModel:
    def test_generate_receives_the_configured_temperature(self, monkeypatch):
        """設定進了鍵卻沒傳給模型的話，鍵是對的而行為沒變——**更難發現**。

        （`generate` 是在 `call_model` 內部 import 的，所以要 patch
        來源模組 `weft.stages.providers`，不是 `understand`。）
        """
        from weft.stages import providers, understand

        seen = {}

        def fake_generate(model, system, parts, schema, temperature=None, **kw):
            seen["temperature"] = temperature
            raise RuntimeError("stop here —— 只確認參數有傳到")

        monkeypatch.setattr(providers, "generate", fake_generate)
        cfg = Config().s4
        cfg.temperature = 0.0
        with pytest.raises(RuntimeError, match="stop here"):
            understand.call_model([_segment()], {}, None, cfg)
        assert seen["temperature"] == 0.0

    def test_the_default_is_still_the_historical_value(self):
        """0.2 是 2026-08-13 之前寫死的值。改預設值等於改所有既有資料的
        產生條件——那要有量測撐著，不能順手改。"""
        assert Config().s4.temperature == 0.2
