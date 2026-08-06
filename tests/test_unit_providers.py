"""模型供應者抽象層（SDD §2.3，v0.4）。"""

from __future__ import annotations

import pytest

from weft.stages.providers import Part, UnknownProvider, costs_quota, split_spec


def test_spec_is_provider_colon_model():
    assert split_spec("gemini:gemini-3.1-flash-lite") == ("gemini", "gemini-3.1-flash-lite")
    # ollama 的模型名本身含冒號，只切第一個
    assert split_spec("ollama:qwen2.5vl:7b") == ("ollama", "qwen2.5vl:7b")


@pytest.mark.parametrize("bad", ["gemma4:12b", "gemini", "", ":x", "x:"])
def test_provider_must_be_explicit(bad):
    """§5.5 #6：本地 fallback 要是明確的設定開關。

    「猜猜看這是本地還是雲端」會讓那條規定失效——`gemma4:12b` 看起來
    像個模型名，但沒說是哪個供應者，必須報錯而不是猜。
    """
    with pytest.raises(UnknownProvider):
        split_spec(bad)


def test_only_gemini_costs_quota():
    assert costs_quota("gemini:gemini-3.1-flash-lite") is True
    assert costs_quota("ollama:qwen2.5vl:7b") is False


def test_part_must_be_text_xor_image():
    Part(text="x")
    Part(image=b"\x89PNG")
    with pytest.raises(ValueError):
        Part()
    with pytest.raises(ValueError):
        Part(text="x", image=b"y")


def test_unknown_provider_is_rejected():
    from weft.stages.providers import generate

    with pytest.raises(UnknownProvider):
        generate("openai:gpt-4", "sys", [Part(text="hi")], {})


# ---------------------------------------------------------------------------
# 本地模型不得佔用雲端額度
#
# 實測抓到的真實 bug：全本地配置跑到一半，被自己的額度帳本判定
# 「額度用盡」而停下——`s4_understand` 無條件記帳。
# ---------------------------------------------------------------------------

def test_local_stages_do_not_consume_quota(tmp_path, monkeypatch):
    from weft.config import Config
    from weft.paths import OutPaths, WorkPaths
    from weft.quota import QuotaLedger
    from weft.stages import cloud
    from weft.stages.understand import BatchResult
    from tests.factories import make_ir

    cfg = Config()
    cfg.out_dir = tmp_path / "out"
    cfg.s4.model = "ollama:qwen2.5:14b"
    cfg.s4.batch_segments = 1
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)

    work = WorkPaths(tmp_path / "work", "vid")
    work.dir.mkdir(parents=True, exist_ok=True)
    ir = make_ir(work.dir)

    monkeypatch.setattr(cloud, "_load_cached", lambda *a, **k: None)
    monkeypatch.setattr(
        "weft.stages.understand.call_model",
        lambda *a, **k: BatchResult(
            per_segment={s.segment_id: {"segment_id": s.segment_id, "corrections": [],
                                        "summary": "x", "content_blocks": [], "terms": []}
                         for s in ir.segments},
            input_tokens=100, output_tokens=50, model_used="ollama:qwen2.5:14b"),
    )

    cloud.s4_understand(cfg, work, ir.segments, ir.slides, None)

    ledger = QuotaLedger(OutPaths(cfg.out_dir).quota_db, cfg.quota)
    assert ledger.usage_today("ollama:qwen2.5:14b").requests == 0, "本地模型不得記進額度帳本"
