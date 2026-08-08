"""SDD §5.5 禁止捷徑清單的機械式護欄。

§5.5 的規定寫在文件裡，人（或 AI coding agent）看不看得到、看了守不守，
沒有保證。能自動檢查的就自動檢查——這些測試把「別走捷徑」從自律變成紅燈。

涵蓋不到的（例如「不得只用逐字稿讓 LLM 猜投影片內容」）記在
docs/known-risks.md，靠 code review 與 §5.6 的人工抽檢。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "weft"
TESTS = REPO / "tests"


def python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def collect_test_files() -> list[Path]:
    return sorted(TESTS.glob("test_*.py"))


# --------------------------------------------------------------------------
# §5.5 #10：e2e 不得用 mock；單元測試可 mock，但檔名須含 `_unit_`
# --------------------------------------------------------------------------

_MOCK_HINTS = re.compile(
    r"\b(unittest\.mock|from\s+unittest\s+import\s+mock|monkeypatch|MagicMock|"
    r"mocker\.|patch\(|@patch)",
)


def test_only_unit_files_may_mock():
    """§5.5 #10：單元測試可以 mock，但**檔名須含 `_unit_`**。"""
    offenders = []
    for path in collect_test_files():
        if "_unit_" in path.name:
            continue
        source = path.read_text(encoding="utf-8")
        # 剝掉字串與註解後再找，避免誤判「文中提到 mock」的說明文字
        body = _strip_literals_and_comments(source)
        if _MOCK_HINTS.search(body):
            offenders.append(path.name)
    assert not offenders, (
        f"這些非 _unit_ 測試檔用了 mock，違反 SDD §5.5 #10：{offenders}。"
        "若確實只是單元測試，請把檔名改成含 `_unit_`。"
    )


def test_e2e_tests_exist_and_are_not_mocked():
    """§5.5 #10：e2e 測試須獨立存在，且不得用 mock 取代真實模型呼叫。"""
    e2e = [p for p in collect_test_files() if "e2e" in p.name]
    assert e2e, "找不到 e2e 測試檔。SDD §7.1 要求 Phase 0 就要有（可跑不通，但要存在）。"
    for path in e2e:
        assert "_unit_" not in path.name, f"{path.name} 同時標了 e2e 與 _unit_"
        body = _strip_literals_and_comments(path.read_text(encoding="utf-8"))
        assert not _MOCK_HINTS.search(body), f"{path.name} 用了 mock"


def _strip_literals_and_comments(source: str) -> str:
    """移除字串常值與註解。讓「在 docstring 裡討論 mock」不會被誤判。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.end_lineno is not None:
                spans.append((node.lineno, node.end_lineno))
    drop = {ln for a, b in spans for ln in range(a, b + 1)}
    lines = [
        "" if i + 1 in drop else line.split("#")[0]
        for i, line in enumerate(source.splitlines())
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# §5.5 #11：不得用「跑過就好」的斷言取代量化門檻
# --------------------------------------------------------------------------

_WEAK_ASSERT = re.compile(
    r"assert\s+\w+\s+is\s+not\s+None\s*$|assert\s+True\s*$|assert\s+1\s*$",
    re.MULTILINE,
)


def test_no_smoke_test_assertions_in_metric_tests():
    """§5.5 #11：不得在測試中用 `assert result is not None` 取代量化門檻。

    範圍限於會驗收 §5.2 指標的檔案——結構性測試用 is not None 是合理的。
    """
    offenders = []
    for path in collect_test_files():
        source = path.read_text(encoding="utf-8")
        if "thresholds" not in source:
            continue
        # **逐行掃描，不做位移換算。** 原本是拿去除後字串的位移去索引原檔，
        # 但去除字串與註解會改變長度，回報的行號因此一直是錯的——
        # 錯誤訊息指到不相干的那一行，反而讓人往錯的地方查。
        # 逐行比對沒有位移，也就沒有對錯的問題。
        for i, line in enumerate(_strip_literals_and_comments(source).splitlines(), 1):
            if _WEAK_ASSERT.search(line):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"量化門檻測試中出現「跑過就好」的斷言：{offenders}"


# --------------------------------------------------------------------------
# §5.5 #7：門檻不得可調
# --------------------------------------------------------------------------


def test_thresholds_module_has_no_imports_from_config():
    """門檻若能讀設定，就等於可以被覆寫。"""
    source = (SRC / "validation" / "thresholds.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            module = getattr(node, "module", "") or ""
            assert "config" not in module, f"thresholds.py 不得 import config（{names}）"
            assert "os" not in names and "environ" not in module, (
                "thresholds.py 不得讀環境變數"
            )


def test_thresholds_are_module_level_constants():
    """門檻必須是模組層級常數，不能是函式回傳值——後者可以被 patch。"""
    tree = ast.parse((SRC / "validation" / "thresholds.py").read_text(encoding="utf-8"))
    functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert not functions, f"thresholds.py 不該有函式或類別：{functions}"


def test_threshold_names_appear_only_in_thresholds_module():
    """防止有人在別處重新定義一份「本地門檻」繞過。"""
    import weft.validation.thresholds as T

    names = [n for n in dir(T) if n.isupper() and n != "ACCEPTANCE_THRESHOLDS"]
    offenders = []
    for path in python_files(SRC):
        if path.name == "thresholds.py":
            continue
        source = path.read_text(encoding="utf-8")
        for name in names:
            if re.search(rf"^{name}\s*[:=]", source, re.MULTILINE):
                offenders.append(f"{path.name}:{name}")
    assert not offenders, f"門檻名稱在他處被重新賦值：{offenders}"


# --------------------------------------------------------------------------
# §5.5 #9：不變量不得降級為 warning
# --------------------------------------------------------------------------


def test_invariants_never_log_instead_of_raising():
    """檢查 invariants.py 沒有引入 logging——一旦開始 log，下一步就是不 raise。"""
    source = (SRC / "validation" / "invariants.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            names = [a.name for a in node.names]
            assert "logging" not in module and "logging" not in names, (
                "invariants.py 不得 import logging（§5.5 #9：不得把 assert 改成 warning 或 log）"
            )
    assert "warnings.warn" not in source


def test_assert_all_raises_unconditionally():
    """assert_all 的函式體必須含 raise，且沒有任何跳過條件。"""
    source = (SRC / "validation" / "invariants.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "assert_all"
    )
    assert any(isinstance(n, ast.Raise) for n in ast.walk(fn))


# --------------------------------------------------------------------------
# §5.5 #12：不得接入訂閱額度
# --------------------------------------------------------------------------

_SUBSCRIPTION_HINTS = re.compile(
    r"(gemini\.google\.com|aistudio\.google\.com/app|oauth2\.googleapis|"
    r"AI\s*Pro|AI\s*Ultra|__Secure-1PSID|SAPISID)",
    re.IGNORECASE,
)


def test_no_subscription_quota_backdoor():
    """§5.5 #12：不得以任何方式把 AI Pro/Ultra 的訂閱額度接入程式化呼叫。
    額度來源固定為 Gemini API key。"""
    offenders = []
    for path in python_files(SRC):
        body = _strip_literals_and_comments(path.read_text(encoding="utf-8"))
        if _SUBSCRIPTION_HINTS.search(body):
            offenders.append(path.name)
    assert not offenders, f"疑似繞道訂閱額度：{offenders}"


# --------------------------------------------------------------------------
# §5.5 #6：本地 fallback 必須是明確開關且預設關閉
# --------------------------------------------------------------------------


def test_local_fallback_is_opt_in_and_defaults_off():
    """額度耗盡時不得**靜默**改用本地模型。"""
    from weft.config import S4Config

    cfg = S4Config()
    assert hasattr(cfg, "allow_local_fallback")
    assert cfg.allow_local_fallback is False, "本地 fallback 預設必須關閉"


def test_understanding_can_record_which_model_was_used():
    """§5.5 #6：本地 fallback 的輸出須標記 model_used。"""
    from weft.ir import Understanding

    assert "model_used" in Understanding.model_fields


# --------------------------------------------------------------------------
# §5.5 #13：不得靠撞 429 探測額度
# --------------------------------------------------------------------------


def test_quota_config_defines_active_throttling():
    """必須主動用 quota ledger 預估與節流。"""
    from weft.config import QuotaConfig

    cfg = QuotaConfig()
    assert 0 < cfg.safety_ratio < 1, "safety_ratio 應為配額的安全水位比例"
    assert cfg.reset_timezone == "America/Los_Angeles", (
        "SDD §6.2：RPD 於太平洋時間午夜重置，須用 zoneinfo 計算，不得寫死時差"
    )


def test_no_hardcoded_timezone_offset():
    """§6.2：不要寫死時差。搜尋 UTC+8 / +0800 之類的痕跡。"""
    offenders = []
    for path in python_files(SRC):
        body = _strip_literals_and_comments(path.read_text(encoding="utf-8"))
        if re.search(r"(UTC[+-]\d|timedelta\(hours\s*=\s*-?[78]\b|\+0800)", body):
            offenders.append(path.name)
    assert not offenders, f"疑似寫死時差：{offenders}"


# --------------------------------------------------------------------------
# 契約完整性
# --------------------------------------------------------------------------


def test_content_type_enum_is_closed():
    """§3.4：content_blocks[].type 的封閉列舉。"""
    from weft.ir import ContentType

    assert {t.value for t in ContentType} == {"經文原文", "白話解說", "圖表描述", "口頭延伸"}


def test_ir_models_forbid_extra_fields():
    """階段之間偷加欄位會讓契約悄悄漂移。"""
    import weft.ir as ir_module
    from pydantic import BaseModel

    for name in dir(ir_module):
        obj = getattr(ir_module, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
            if obj.__module__ != ir_module.__name__:
                continue
            assert obj.model_config.get("extra") == "forbid", f"{name} 未設 extra=forbid"


@pytest.mark.parametrize(
    "stage_module,names",
    [
        ("weft.stages.local", ["s0_fetch", "s1a_transcript", "s1b_slides", "s3_align"]),
        ("weft.stages.cloud", ["s4_understand", "s5_synthesize", "s6_render"]),
    ],
)
def test_all_sdd_stages_have_entry_points(stage_module, names):
    """SDD §4 的每個階段都要有對應函式，即使尚未實作。"""
    import importlib

    module = importlib.import_module(stage_module)
    for name in names:
        assert hasattr(module, name), f"{stage_module} 缺少階段 {name}"


# --------------------------------------------------------------------------
# 設定檔與程式預設不得漂移
# --------------------------------------------------------------------------


def test_default_yaml_matches_code_defaults():
    """`configs/default.yaml` 存在是為了讓可調參數一目了然，不是為了覆寫。

    兩邊漂移的後果很安靜：使用者讀 YAML 以為是這樣，程式實際用另一組值。
    這條測試把「改了程式預設就得同步 YAML」變成紅燈。
    """
    from weft.config import Config

    on_disk = Config.load(REPO / "configs" / "default.yaml")
    in_code = Config()

    differences = {
        key: (value, in_code.model_dump()[key])
        for key, value in on_disk.model_dump().items()
        if value != in_code.model_dump()[key]
    }
    assert not differences, f"configs/default.yaml 與程式預設不一致：{differences}"


def test_default_yaml_contains_no_acceptance_thresholds():
    """§5.5 #7：**驗收門檻**不得經設定檔覆寫。

    範圍限於 `ACCEPTANCE_THRESHOLDS`——`MAX_CHUNK_CHARS`（§4.9 的切分規則）
    與 `COVERAGE_TOLERANCE_SEC`（§5.3 的容忍值）是處理參數，可以調。
    """
    from weft.validation.thresholds import ACCEPTANCE_THRESHOLDS

    body = (REPO / "configs" / "default.yaml").read_text(encoding="utf-8").lower()
    for name in ACCEPTANCE_THRESHOLDS:
        assert name.lower() not in body, f"設定檔出現了驗收門檻 {name}"


def test_acceptance_threshold_list_is_complete():
    """新增 §5.2 門檻卻忘了登記到 ACCEPTANCE_THRESHOLDS，護欄就會漏掉它。"""
    import weft.validation.thresholds as T

    declared = set(T.ACCEPTANCE_THRESHOLDS)
    processing = {"MAX_CHUNK_CHARS", "COVERAGE_TOLERANCE_SEC",
                  "ACCEPTANCE_THRESHOLDS",
                  # 只記錄不設門檻的量測項（票 03）。它們不是驗收門檻，
                  # 但也不是處理參數——列在這裡是為了讓下面的差集算得對。
                  "OBSERVED_ONLY"}
    actual = {n for n in dir(T) if n.isupper() and not n.startswith("_")}
    assert actual - processing == declared, (
        f"未登記的門檻：{actual - processing - declared}；"
        f"登記了但不存在：{declared - actual}"
    )


# --------------------------------------------------------------------------
# 門檻不得懸空（R24）
#
# §5.5 #7 防的是「有人偷偷調門檻」，防不了「門檻根本沒接上任何驗收」。
# 盤點（experiments/r24_threshold_audit/）發現 10 項門檻只有 3 項真的在
# 拿實際輸出比對；其餘幾項只在 test_unit_metrics.py 有「常數值斷言」
# （`assert T.X == 0.90`）——那條會綠燈，看起來像門檻有在管，
# 但它只證明了常數沒被改過。
# --------------------------------------------------------------------------

#: 只驗常數值本身的斷言，例如 `assert T.BOUNDARY_F1_REAL == 0.75`。
#: 這種斷言**不算**把門檻接上驗收。
_CONSTANT_ASSERT = re.compile(
    r"assert\s+\w+\.([A-Z_]+)\s*(==|is)\s*(?:[\d.]+|None|True|False)\s*(?:#.*)?$",
    re.MULTILINE,
)

#: 已知尚未接上、且**已在文件中記錄為欠帳**的門檻。
#: 加進這裡必須同時在 docs/known-risks.md 留下對應條目——
#: 這個清單是「明知故犯的紀錄」，不是豁免權。
DANGLING_THRESHOLDS: frozenset[str] = frozenset({
    "TERM_CORRECTION_RECALL",       # 值為 None，語意是「量測但不 assert」
    "ALIGNMENT_MEDIAN_ERROR_SEC",   # v0.3 移除語意吸附後量什麼待決，見 R24
})


def test_every_acceptance_threshold_is_actually_enforced():
    """每個 §5.2 門檻都必須被至少一個**非常數斷言**使用。

    「常數值斷言」不算——`assert T.X == 0.90` 只證明常數沒被改過，
    與「有沒有拿真實輸出去比」是兩回事。
    """
    from weft.validation import thresholds as T

    used: set[str] = set()
    for path in collect_test_files():
        source = _strip_literals_and_comments(path.read_text(encoding="utf-8"))
        constant_only = {m.group(1) for m in _CONSTANT_ASSERT.finditer(source)}
        for name in T.ACCEPTANCE_THRESHOLDS:
            if name not in source:
                continue
            # 出現次數若全部來自常數斷言，就不算接上
            occurrences = source.count(name)
            if occurrences > (1 if name in constant_only else 0):
                used.add(name)

    dangling = set(T.ACCEPTANCE_THRESHOLDS) - used - DANGLING_THRESHOLDS
    assert not dangling, (
        f"這些門檻沒有被任何實際驗收使用（只有常數斷言或完全沒用到）：{sorted(dangling)}。"
        "門檻懸空時測試會全綠，但實際上什麼都沒在管。"
        "若確定暫時無法接上，加進 DANGLING_THRESHOLDS 並在 known-risks 留紀錄。"
    )


def test_dangling_thresholds_are_documented():
    """明知故犯的清單必須在 known-risks.md 有對應紀錄。"""
    risks = (REPO / "docs" / "known-risks.md").read_text(encoding="utf-8")
    missing = [name for name in DANGLING_THRESHOLDS if name not in risks]
    assert not missing, (
        f"{missing} 列在 DANGLING_THRESHOLDS 卻沒有在 docs/known-risks.md 說明。"
        "這個清單是紀錄，不是豁免權。"
    )
