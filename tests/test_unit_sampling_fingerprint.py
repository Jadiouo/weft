"""取樣指紋涵蓋所有取樣參數（2026-08-13）。

**這個檔案存在的理由是：不要有第七次。**

| | 冪等鍵漏掉的東西 |
|---|---|
| D20 / D22 | 早期的階段參數 |
| D30 | `prompt_version` |
| D32 | `segment_id` 的位置性 |
| `DEPTH_ALPHA` | 模組常數不在 `cfg.s3` 裡 |
| `temperature` | 寫死在 `providers.generate()` 的預設值 |

前六次都是「加了一個決定結果的東西，忘了加進比對邏輯」。
逐個欄位比對要求每次都記得——**那個「記得」遲早會失效**，
而失效的症狀是「改了沒效果」，最難察覺。

改成單一指紋之後，**這裡用反射檢查沒有欄位被漏掉**：
在 `S4Config` 加一個取樣參數卻沒登記，`test_no_sampling_field_is_missed`
就會紅。
"""

from __future__ import annotations

import pytest

from weft.config import Config
from weft.stages.cloud import SAMPLING_PARAMS, sampling_fingerprint


class TestEveryParamChangesIt:
    @pytest.mark.parametrize(("name", "value"), [
        ("temperature", 0.9),
        ("seed", 42),
        ("top_k", 1),
    ])
    def test_changing_a_param_changes_the_fingerprint(self, name, value):
        a = Config().s4
        b = Config().s4
        setattr(b, name, value)
        assert getattr(a, name) != value, f"{name} 的測試值與預設相同，這條測不到東西"
        assert sampling_fingerprint(a) != sampling_fingerprint(b)

    def test_same_config_gives_the_same_fingerprint(self):
        assert sampling_fingerprint(Config().s4) == sampling_fingerprint(Config().s4)

    def test_unrelated_field_does_not_change_it(self):
        """不影響取樣的東西不該讓快取失效——那會白白重跑幾小時。"""
        a, b = Config().s4, Config().s4
        b.batch_segments = a.batch_segments + 1
        assert sampling_fingerprint(a) == sampling_fingerprint(b)


class TestNothingIsMissed:
    """**這條才是防第七次的那條。**"""

    #: `S4Config` 裡**不影響取樣結果**的欄位。新增欄位時要嘛加進
    #: `SAMPLING_PARAMS`、要嘛加進這裡，**逼人做一次判斷**。
    NOT_SAMPLING = frozenset({
        "model", "prompt_version", "batch_segments", "prev_summary_max_chars",
        "max_retries", "send_images", "num_ctx", "enabled",
        "description_checker_model", "provenance",
        # 重試與退避：影響**要不要再打一次**，不影響單次呼叫的取樣分佈
        "retry_backoff_sec",
        # 本地備援：換模型的話 `model_used` 那一關就會擋下快取，
        # 不需要指紋再管一次
        "allow_local_fallback", "local_fallback_model",
    })

    def test_no_sampling_field_is_missed(self):
        fields = set(type(Config().s4).model_fields)
        unclassified = fields - set(SAMPLING_PARAMS) - self.NOT_SAMPLING
        assert not unclassified, (
            f"`S4Config` 有沒分類的欄位：{sorted(unclassified)}。\n"
            f"**它影響取樣嗎？** 影響 → 加進 `cloud.SAMPLING_PARAMS`；"
            f"不影響 → 加進本測試的 `NOT_SAMPLING`。\n"
            f"這個 repo 已經六次因為冪等鍵漏東西而量出假結論，"
            f"所以這裡強迫做一次判斷，不讓它預設被忽略。"
        )

    def test_the_registered_params_actually_exist(self):
        """登記了但欄位不存在 → 指紋永遠拿到 None，等於沒涵蓋。"""
        cfg = Config().s4
        missing = [n for n in SAMPLING_PARAMS if not hasattr(cfg, n)]
        assert not missing, f"`SAMPLING_PARAMS` 裡有不存在的欄位：{missing}"


class TestItIsUsedAsTheCacheKey:
    def test_cache_with_a_different_fingerprint_is_rejected(self, tmp_path,
                                                            monkeypatch):
        from weft.ir import Understanding
        from weft.paths import WorkPaths
        from weft.stages import cloud

        cfg = Config().s4
        cfg.model = "ollama:qwen2.5:14b"
        work = WorkPaths(tmp_path / "work", "vid")
        work.ensure_dirs()
        monkeypatch.setattr(cloud, "_index_of", lambda *_a: 0)
        monkeypatch.setattr(cloud, "segment_fingerprint", lambda *_a: "fp")

        from weft.ir import Segment, SegmentMode

        seg = Segment(segment_id="vid#000", video_id="vid", t_start=0.0,
                      t_end=60.0, mode=SegmentMode.SPEAKER_ONLY,
                      boundary_method="topic_shift", cue_indices=[0])

        def write(fp):
            work.understanding(0).write_text(
                Understanding(summary="s", model_used=cfg.model,
                              prompt_version=cfg.prompt_version,
                              input_fingerprint="fp",
                              sampling_fingerprint=fp).model_dump_json(),
                encoding="utf-8")

        write(sampling_fingerprint(cfg))
        assert cloud._load_cached(work, seg, cfg) is not None, "相同指紋應命中"

        write("deadbeefdeadbeef")
        assert cloud._load_cached(work, seg, cfg) is None, "不同指紋不得命中"

        write(None)
        assert cloud._load_cached(work, seg, cfg) is None, (
            "舊快取沒有指紋（None）時要**保守重跑**——"
            "讓它命中等於相信一個無法驗證的假設（D32 就是這樣出事的）"
        )
