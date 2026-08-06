"""S1a 的簡繁轉換（D24）。

實測依據（R18）：Whisper large-v3 對這批繁體素材輸出 9.43% 簡體字，
§5.4 溯源通過率因此從 97.2% 掉到 91.5%。
"""

from __future__ import annotations

from weft.stages.transcribe import to_traditional


def test_simplified_asr_output_is_converted():
    rows = [(0.0, 1.0, "这个说法没有对"), (1.0, 2.0, "阿赖耶识的种子")]
    out = to_traditional(rows, "s2tw")
    assert [t for _a, _b, t in out] == ["這個說法沒有對", "阿賴耶識的種子"]


def test_conversion_is_a_noop_on_traditional_text():
    """手動字幕本來就是繁體，轉換不得改動任何字。"""
    rows = [(0.0, 1.0, "太上老君內觀經"), (1.0, 2.0, "五月五行分藏，以安神也。")]
    assert to_traditional(rows, "s2tw") == rows


def test_timestamps_are_preserved():
    rows = [(1.5, 9.25, "这个")]
    out = to_traditional(rows, "s2tw")
    assert [(a, b) for a, b, _t in out] == [(1.5, 9.25)]


def test_none_disables_conversion():
    rows = [(0.0, 1.0, "这个说法")]
    assert to_traditional(rows, None) == rows


def test_taiwan_character_forms_not_mainland_ones():
    """用 s2tw 而非 s2t：s2t 產出大陸標準字形，與台灣素材對不起來。

    實測 R17 的 Whisper 輸出，s2t 給 `爲`×71、`裏`×35、`纔`×5、`着`×5、`喫`×3。
    """
    rows = [(0.0, 1.0, "因为这里才着手吃饭")]
    assert to_traditional(rows, "s2tw")[0][2] == "因為這裡才著手吃飯"
    assert to_traditional(rows, "s2t")[0][2] == "因爲這裏才着手喫飯"


def test_vocabulary_is_not_substituted():
    """用 s2tw 而非 s2twp：後者會改詞彙，那是改語意不是轉字集。

    實測 s2twp 把 `局部`→`區域性`、`運`→`執`、`序`→`式`。
    這批素材談的是人體的「局部」，換掉會改變意思。
    """
    rows = [(0.0, 1.0, "只见微观局部")]
    assert to_traditional(rows, "s2tw")[0][2] == "只見微觀局部"
    assert to_traditional(rows, "s2twp")[0][2] == "只見微觀區域性"


def test_the_real_whisper_errors_from_r18_are_fixed_by_conversion():
    """R18 對齊出來的實際簡體輸出。"""
    rows = [(0.0, 1.0, "三月阳神为三魂四月阴灵为七魄")]
    assert to_traditional(rows, "s2tw")[0][2] == "三月陽神為三魂四月陰靈為七魄"
