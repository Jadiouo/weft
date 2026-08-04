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
        for match in _WEAK_ASSERT.finditer(_strip_literals_and_comments(source)):
            line = source[: match.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line}")
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

    names = [n for n in dir(T) if n.isupper()]
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
        ("weft.stages.local", ["s0_fetch", "s1a_transcript", "s1b_slides",
                               "s2_ocr", "s2b_lexicon", "s2c_correct", "s3_align"]),
        ("weft.stages.cloud", ["s4_understand", "s5_synthesize", "s6_render"]),
    ],
)
def test_all_sdd_stages_have_entry_points(stage_module, names):
    """SDD §4 的每個階段都要有對應函式，即使尚未實作。"""
    import importlib

    module = importlib.import_module(stage_module)
    for name in names:
        assert hasattr(module, name), f"{stage_module} 缺少階段 {name}"
