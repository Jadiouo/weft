"""續跑語意的測試。SDD §6.3。

「批次跑數十支影片時，中途失敗是常態而非例外。」（§2.1 原則四）

作廢邏輯錯了的後果很難察覺：不是崩潰，而是**沿用了本該重算的舊資料**。
所以規則要逐條釘死：參數 hash 變更 → 該階段及其下游需重跑；**上游不動**。
"""

from __future__ import annotations

import pytest

from weft.config import Config
from weft.state import (
    DEPENDENCIES,
    PREPARE_STAGES,
    UNDERSTAND_STAGES,
    Stage,
    StageStatus,
    VideoState,
    downstream_of,
    params_hash,
)


# --------------------------------------------------------------------------
# 依賴圖與 SDD §2.2 的資料流一致
# --------------------------------------------------------------------------


def test_every_stage_has_declared_dependencies():
    assert set(DEPENDENCIES) == set(Stage)


def test_dependency_graph_is_acyclic():
    for stage in Stage:
        assert stage not in downstream_of(stage), f"{stage} 出現在自己的下游"


def test_s0_is_the_only_root():
    roots = [s for s, deps in DEPENDENCIES.items() if not deps]
    assert roots == [Stage.S0_FETCH]


def test_s3_depends_on_both_transcript_and_candidates():
    """§4.6：對齊需要逐字稿與候選區段兩者。

    v0.4 起候選那一側是 **S1c**（去重後）而非 S1b——S3 指派逐字稿時
    要用去重後的結果，否則同一張投影片的多次出現會被當成不同的段落來源。
    """
    assert set(DEPENDENCIES[Stage.S3_ALIGN]) == {Stage.S1A_TRANSCRIPT, Stage.S1C_DEDUP}
    assert set(DEPENDENCIES[Stage.S1C_DEDUP]) == {Stage.S1B_SLIDES}


def test_local_ocr_chain_is_gone():
    """v0.3 移除了 S2／S2b／S2c（本地 OCR + 詞庫 + 術語校正）。

    這條測試釘住「不要為了備用把它加回來」——留著就會有人去修它。
    術語校正改由 S4 完成。

    **v0.4 新增 S1c（投影片去重）**——它是純 CV，與被移除的 OCR 鏈無關：
    不讀文字、不需詞庫、不做繁簡轉換。加它進來不違反本測試的用意。
    """
    assert {s.value for s in Stage} == {"S0", "S1a", "S1b", "S1c", "S3", "S4", "S5", "S6"}
    # 被移除的三個階段不得以任何形式回來
    assert not {"S2", "S2b", "S2c"} & {s.value for s in Stage}


def test_prepare_and_understand_partition_all_stages():
    """§6.4 producer/consumer 分離：兩個入口合起來要涵蓋全部階段且不重疊。"""
    assert set(PREPARE_STAGES) | set(UNDERSTAND_STAGES) == set(Stage)
    assert not set(PREPARE_STAGES) & set(UNDERSTAND_STAGES)


def test_prepare_stages_do_not_cost_quota():
    """§2.1 原則三：S0–S3 是 producer，不花額度。S4 才是 consumer。"""
    assert Stage.S4_UNDERSTAND not in PREPARE_STAGES
    assert Stage.S4_UNDERSTAND in UNDERSTAND_STAGES


# --------------------------------------------------------------------------
# 下游計算
# --------------------------------------------------------------------------


def test_downstream_is_transitive():
    """S1b 變更要一路作廢到 S6，不是只作廢直接的下一階段。"""
    assert Stage.S6_RENDER in downstream_of(Stage.S1B_SLIDES)


def test_downstream_of_s1b_reaches_alignment_and_beyond():
    ds = downstream_of(Stage.S1B_SLIDES)
    assert {Stage.S3_ALIGN, Stage.S4_UNDERSTAND, Stage.S6_RENDER} <= ds


def test_downstream_of_last_stage_is_empty():
    assert downstream_of(Stage.S6_RENDER) == set()


def test_s1a_change_does_not_invalidate_s1b():
    """兩者都只依賴 S0，是平行分支。重跑 Whisper 不該讓抽幀白費。"""
    assert Stage.S1B_SLIDES not in downstream_of(Stage.S1A_TRANSCRIPT)


def test_s1b_change_does_not_invalidate_s1a():
    assert Stage.S1A_TRANSCRIPT not in downstream_of(Stage.S1B_SLIDES)


# --------------------------------------------------------------------------
# 參數 hash
# --------------------------------------------------------------------------


def test_params_hash_is_stable_across_key_order():
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})


def test_params_hash_changes_with_value():
    assert params_hash({"fps": 1.0}) != params_hash({"fps": 2.0})


def test_params_hash_distinguishes_int_and_float():
    """1 與 1.0 在 fps 這種參數上可能導致不同行為，不該被視為同一組參數。"""
    assert params_hash({"fps": 1}) != params_hash({"fps": 1.0})


def test_each_stage_config_has_its_own_hash():
    cfg = Config()
    hashes = {
        cfg.s0.params_hash(), cfg.s1a.params_hash(), cfg.s1b.params_hash(),
        cfg.s3.params_hash(), cfg.s4.params_hash(), cfg.s5.params_hash(),
        cfg.s6.params_hash(),
    }
    assert len(hashes) == 7, "不同階段的設定產生了相同的 hash"


def test_changing_one_stage_config_does_not_affect_others():
    cfg = Config()
    before = cfg.s1a.params_hash()
    cfg.s1b.fps = 2.0
    assert cfg.s1a.params_hash() == before


# --------------------------------------------------------------------------
# 狀態轉移
# --------------------------------------------------------------------------


def test_is_satisfied_requires_both_done_and_matching_hash():
    state = VideoState(video_id="v")
    state.mark_done(Stage.S1B_SLIDES, "hash_a")

    assert state.is_satisfied(Stage.S1B_SLIDES, "hash_a")
    assert not state.is_satisfied(Stage.S1B_SLIDES, "hash_b")  # 參數變了
    assert not state.is_satisfied(Stage.S3_ALIGN, "hash_a")  # 沒跑過


def test_failed_stage_is_not_satisfied():
    state = VideoState(video_id="v")
    state.mark_failed(Stage.S1A_TRANSCRIPT, "Whisper OOM")
    assert not state.is_satisfied(Stage.S1A_TRANSCRIPT, "any")
    assert state.stages[Stage.S1A_TRANSCRIPT].error == "Whisper OOM"


def test_mark_done_clears_previous_error():
    state = VideoState(video_id="v")
    state.mark_failed(Stage.S1A_TRANSCRIPT, "OOM")
    state.mark_done(Stage.S1A_TRANSCRIPT, "h")
    assert state.stages[Stage.S1A_TRANSCRIPT].error is None


def test_invalidate_downstream_leaves_upstream_untouched():
    """SDD §6.3 的核心規則：作廢下游，**上游不動**。"""
    state = VideoState(video_id="v")
    for stage in Stage:
        state.mark_done(stage, "h")

    state.invalidate_downstream(Stage.S1B_SLIDES)

    # 上游仍完成
    assert state.is_satisfied(Stage.S0_FETCH, "h")
    # 平行分支不受影響
    assert state.is_satisfied(Stage.S1A_TRANSCRIPT, "h")
    # 自己也不動——變的是它的參數，重跑由呼叫端決定
    assert state.is_satisfied(Stage.S1B_SLIDES, "h")
    # 下游全部作廢
    for stage in (Stage.S3_ALIGN, Stage.S4_UNDERSTAND,
                  Stage.S5_SYNTHESIZE, Stage.S6_RENDER):
        assert not state.is_satisfied(stage, "h"), f"{stage} 未被作廢"


def test_invalidating_s4_clears_segment_level_checkpoints():
    """§6.3：S4 以 segment 為粒度斷點。作廢 S4 卻留著已完成清單，
    會讓下次執行以為那些 segment 還有效——用舊 prompt 的結果混進新一輪。"""
    state = VideoState(video_id="v")
    state.mark_done(Stage.S4_UNDERSTAND, "h")
    state.understood_segments = ["v#000", "v#001"]

    state.invalidate_downstream(Stage.S3_ALIGN)
    assert state.understood_segments == []


def test_invalidating_late_stage_keeps_s4_checkpoints():
    """作廢 S6 不該讓已經花掉額度的 S4 結果消失。"""
    state = VideoState(video_id="v")
    state.mark_done(Stage.S4_UNDERSTAND, "h")
    state.understood_segments = ["v#000"]

    state.invalidate_downstream(Stage.S5_SYNTHESIZE)
    assert state.understood_segments == ["v#000"]


# --------------------------------------------------------------------------
# 持久化
# --------------------------------------------------------------------------


def test_state_roundtrips_through_disk(tmp_path):
    path = tmp_path / "state.json"
    state = VideoState(video_id="v")
    state.mark_done(Stage.S0_FETCH, "h0")
    state.understood_segments = ["v#000"]
    state.save(path)

    loaded = VideoState.load(path)
    assert loaded.is_satisfied(Stage.S0_FETCH, "h0")
    assert loaded.understood_segments == ["v#000"]


def test_save_is_atomic(tmp_path):
    """批次跑到一半被 kill，不該留下半個 state.json——那會讓整支影片重跑。"""
    path = tmp_path / "state.json"
    VideoState(video_id="v").save(path)
    assert not list(tmp_path.glob("*.tmp"))
    assert VideoState.load(path).video_id == "v"


def test_load_or_new_creates_when_absent(tmp_path):
    state = VideoState.load_or_new(tmp_path / "nope.json", "v")
    assert state.video_id == "v"
    assert state.stages == {}


def test_load_raises_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoState.load(tmp_path / "nope.json")


def test_unknown_fields_are_rejected(tmp_path):
    """state.json 手動改壞時要炸得明顯，不要默默忽略。"""
    from pydantic import ValidationError

    path = tmp_path / "state.json"
    path.write_text('{"video_id": "v", "stages": {}, "surprise": 1}', encoding="utf-8")
    with pytest.raises(ValidationError):
        VideoState.load(path)


# --------------------------------------------------------------------------
# 編排層的 sync_state
# --------------------------------------------------------------------------


def test_sync_state_detects_config_drift(tmp_path):
    """設定改了、work/ 還留著舊產物——這是續跑最常見的情境。"""
    from weft.pipeline import stage_params, sync_state

    cfg = Config()
    state = VideoState(video_id="v")
    for stage in Stage:
        state.mark_done(stage, stage_params(cfg, stage))

    assert sync_state(cfg, state) == set()  # 沒有變更

    cfg.s1b.fps = 4.0
    affected = sync_state(cfg, state)

    assert Stage.S3_ALIGN in affected
    assert Stage.S6_RENDER in affected
    assert Stage.S1A_TRANSCRIPT not in affected  # 平行分支，不受影響
    assert state.is_satisfied(Stage.S0_FETCH, stage_params(cfg, Stage.S0_FETCH))


def test_sync_state_is_idempotent(tmp_path):
    from weft.pipeline import stage_params, sync_state

    cfg = Config()
    state = VideoState(video_id="v")
    for stage in Stage:
        state.mark_done(stage, stage_params(cfg, stage))

    cfg.s1b.fps = 2.0
    first = sync_state(cfg, state)
    second = sync_state(cfg, state)

    assert first
    assert second == set(), "第二次同步不該再作廢任何東西"


def test_load_tolerates_stages_removed_by_a_pipeline_revision(tmp_path):
    """管線改版後，舊的 state.json 會留著已不存在的階段名。

    直接 validate 會讓使用者**整個 work/ 目錄無法載入**——等於強迫他重跑
    幾十支影片的抽幀。v0.3 移除 S2／S2b／S2c 時真的踩到這個。
    """
    import json

    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "video_id": "v",
        "stages": {
            "S0": {"status": "done", "params_hash": "h"},
            "S2": {"status": "done", "params_hash": "h"},      # v0.3 已移除
            "S2b": {"status": "done", "params_hash": "h"},     # 同上
            "S3": {"status": "done", "params_hash": "h"},
        },
        "understood_segments": [],
    }), encoding="utf-8")

    state = VideoState.load(path)
    assert state.is_satisfied(Stage.S0_FETCH, "h")
    assert state.is_satisfied(Stage.S3_ALIGN, "h")
    assert len(state.stages) == 2, "已移除的階段應被丟棄，不該留在 state 中"


def test_load_still_rejects_genuinely_malformed_state(tmp_path):
    """容忍已移除的階段，不等於容忍任何壞資料。"""
    from pydantic import ValidationError

    path = tmp_path / "state.json"
    path.write_text('{"video_id": "v", "stages": {"S0": {"status": "不是合法狀態"}}}',
                    encoding="utf-8")
    with pytest.raises(ValidationError):
        VideoState.load(path)
