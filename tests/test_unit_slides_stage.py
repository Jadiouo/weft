"""S4a 投影片理解（SDD §4.7a，v0.4）。"""

from __future__ import annotations

import json

import pytest

from weft.config import Config
from weft.ir import Slide
from weft.paths import WorkPaths
from weft.stages.slides import RESPONSE_SCHEMA, apply_to_slide, rehydrate, s4a_understand_slides


def _slide(sid: str, dup: str | None = None) -> Slide:
    s = Slide(slide_id=sid, image_path=f"03_slides/{sid}.png",
              t_first_seen=0.0, t_last_seen=10.0)
    s.duplicate_of = dup
    return s


def test_schema_field_order_forces_transcribe_before_interpret():
    """§4.7a：structured output 逐欄生成，這個順序強制先抄再詮釋。"""
    assert list(RESPONSE_SCHEMA["properties"]) == [
        "is_slide", "reject_reason", "slide_text", "description",
    ]


def test_non_slide_keeps_no_text():
    """判定不是投影片就不留文字——留著會被 §5.4 當成合法的比對來源。"""
    slide = _slide("slide_001")
    apply_to_slide(slide, {"is_slide": False, "reject_reason": "講者鏡頭",
                           "slide_text": "追求法喜的身體健康", "description": "佈景"})
    assert slide.slide_text is None
    assert slide.layout_description is None
    assert slide.reject_reason == "講者鏡頭"


def test_slide_keeps_text_and_description():
    slide = _slide("slide_001")
    apply_to_slide(slide, {"is_slide": True, "reject_reason": "",
                           "slide_text": "太上老君內觀經", "description": "左右綠色方塊"})
    assert slide.slide_text == "太上老君內觀經"
    assert slide.layout_description == "左右綠色方塊"
    assert slide.reject_reason is None


@pytest.fixture
def work(tmp_path):
    w = WorkPaths(tmp_path, "vid")
    (w.dir / "03_slides").mkdir(parents=True, exist_ok=True)
    for sid in ("slide_001", "slide_002"):
        (w.dir / "03_slides" / f"{sid}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return w


def test_only_representatives_are_sent(work, monkeypatch):
    """§4.7a：處理單位是 S1c 去重後的**代表幀**。"""
    seen = []

    def fake(spec, image):
        seen.append(spec)
        return {"is_slide": True, "reject_reason": "", "slide_text": "經文",
                "description": "版面", "model_used": spec, "_tokens": (1, 2)}

    monkeypatch.setattr("weft.stages.slides.understand_slide", fake)
    slides = [_slide("slide_001"), _slide("slide_002", dup="slide_001")]
    stats = s4a_understand_slides(Config(), work, slides)

    assert len(seen) == 1, "被合併的候選幀不該再送一次"
    assert stats["representatives"] == 1


def test_duplicates_inherit_the_representative_text(work, monkeypatch):
    """同一張投影片本來就該有同一份文字——這正是去重的目的。"""
    monkeypatch.setattr("weft.stages.slides.understand_slide",
                        lambda spec, image: {"is_slide": True, "reject_reason": "",
                                             "slide_text": "太上老君內觀經",
                                             "description": "版面",
                                             "model_used": spec, "_tokens": (0, 0)})
    slides = [_slide("slide_001"), _slide("slide_002", dup="slide_001")]
    s4a_understand_slides(Config(), work, slides)
    assert slides[1].slide_text == "太上老君內觀經"


def test_failure_leaves_the_slide_empty_not_partial(work, monkeypatch):
    """§4.7a：仍失敗則該張留空並記錄，**不得以部分輸出充數**。"""
    def boom(spec, image):
        raise RuntimeError("ollama 回傳空內容")

    monkeypatch.setattr("weft.stages.slides.understand_slide", boom)
    slides = [_slide("slide_001")]
    cfg = Config()
    cfg.s4a.max_retries = 0
    cfg.s4a.retry_backoff_sec = 0.0
    stats = s4a_understand_slides(cfg, work, slides)

    assert stats["failed"] == 1
    assert slides[0].slide_text is None


def test_cache_is_keyed_by_model_and_prompt_version(work, monkeypatch):
    calls = []
    monkeypatch.setattr("weft.stages.slides.understand_slide",
                        lambda spec, image: (calls.append(spec) or
                                             {"is_slide": True, "reject_reason": "",
                                              "slide_text": "x", "description": "y",
                                              "model_used": spec, "_tokens": (0, 0)}))
    cfg = Config()
    s4a_understand_slides(cfg, work, [_slide("slide_001")])
    s4a_understand_slides(cfg, work, [_slide("slide_001")])
    assert len(calls) == 1, "同設定第二次應命中快取"

    cfg.s4a.prompt_version = "v2"
    s4a_understand_slides(cfg, work, [_slide("slide_001")])
    assert len(calls) == 2, "改了 prompt_version 就必須重跑"


def test_rehydrate_rebuilds_from_cache(work):
    """續跑時要從快取重建——不重建的話 S4c 拿到空的 slide_context，等於白拆。"""
    work.slide_understanding_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    (work.slide_understanding_dir / "slide_001.json").write_text(json.dumps({
        "is_slide": True, "reject_reason": "", "slide_text": "太上老君內觀經",
        "description": "左右綠色方塊", "model_used": cfg.s4a.model,
        "prompt_version": cfg.s4a.prompt_version,
    }, ensure_ascii=False), encoding="utf-8")

    slides = [_slide("slide_001"), _slide("slide_002", dup="slide_001")]
    assert rehydrate(cfg, work, slides) == 1
    assert slides[0].slide_text == "太上老君內觀經"
    assert slides[1].slide_text == "太上老君內觀經", "被合併的也要跟著重建"
