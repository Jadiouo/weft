"""改了 prompt 就必須改 `prompt_version`——這裡強制它。

§4.7 的冪等鍵是 `segment_id + prompt_version + model`。改了 prompt 卻沒改
版本，舊快取會被當成新結果讀回來：**新 prompt 從未被執行，而量出來的數字
看起來就像「改了沒效果」**。

2026-08-08 真的發生了（D30）。改 S4a 的版本時用了未錨定的字串替換，
`prompt_version: str = "vN"` 在兩個設定類別裡長得一模一樣，S4c 被連帶
bump 了兩次。第二次讓我改完 S4c 的 prompt 後版本**沒有變化**，
快取直接命中，溯源只動了 2 個 block——我差一點就據此下結論說
「改寫來源歸屬的說明沒有用」。

防法：把 prompt 文字的雜湊釘在版本上。改文字 → 雜湊變 → 這個測試紅燈 →
你必須同時改版本與這裡的期望值。**兩件事被綁在一起，就不會只做一半。**
"""

from __future__ import annotations

import hashlib

import pytest

from weft.config import Config


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _prompts() -> dict[str, tuple[str, str]]:
    """`名稱 → (目前的版本字串, 目前的 prompt 雜湊)`。"""
    from weft.stages import cloud, slides, understand

    cfg = Config()
    return {
        "s4a": (
            cfg.s4a.prompt_version,
            _digest(slides.SYSTEM_PROMPT, slides.CLASSIFY_SYSTEM,
                    slides.DESCRIBE_SYSTEM, slides.USER_PROMPT),
        ),
        "s4c": (cfg.s4.prompt_version, _digest(understand.SYSTEM_PROMPT)),
        "s5": (cfg.s5.prompt_version, _digest(cloud.SYNTHESIS_PROMPT)),
    }


#: `階段 → (版本, prompt 雜湊)`。**改 prompt 時兩個都要改。**
#: 只改雜湊不改版本 = 舊快取會被當新結果；只改版本不改雜湊 = 這裡會紅。
EXPECTED: dict[str, tuple[str, str]] = {
    "s4a": ("v5", "fc587b13ef211912"),
    "s4c": ("v7", "d8e608a6b51ed2f8"),
    "s5": ("v1", "c82046a01f1ccf34"),
}


@pytest.mark.parametrize("stage", sorted(EXPECTED))
def test_prompt_change_requires_version_bump(stage: str) -> None:
    version, digest = _prompts()[stage]
    want_version, want_digest = EXPECTED[stage]
    assert (version, digest) == (want_version, want_digest), (
        f"{stage} 的 prompt 或版本變了。\n"
        f"  版本 {want_version} → {version}\n"
        f"  雜湊 {want_digest} → {digest}\n"
        f"**改了 prompt 就要 bump `prompt_version`**，否則舊快取會被當成新結果，"
        f"新 prompt 根本不會執行（D30）。兩個都改完，再把 EXPECTED 更新成新值。"
    )


def test_every_configured_prompt_version_is_covered() -> None:
    """設定裡每一個 `prompt_version` 都要在這裡被釘住。

    漏一個就等於那個階段沒有保護——而這個測試會綠燈，
    製造「有人管」的錯覺（與 R20 盤點出來的那三個空測試同一類）。
    """
    cfg = Config()
    configured = {
        name for name in ("s4a", "s4", "s5")
        if hasattr(getattr(cfg, name), "prompt_version")
    }
    covered = {"s4a", "s4", "s5"} & ({"s4" if s == "s4c" else s for s in EXPECTED})
    assert configured == covered, (
        f"有 `prompt_version` 但沒被釘住的階段：{sorted(configured - covered)}"
    )
