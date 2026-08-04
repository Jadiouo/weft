"""各階段的實作。Phase 0 只有骨架與契約，實作見 SDD §7.2–§7.4。"""

from __future__ import annotations

from typing import NoReturn


class StageNotImplemented(NotImplementedError):
    """尚未實作的階段。

    訊息刻意寫得完整——SDD §7.1 的完成條件是「跑 pytest 會失敗，但失敗
    訊息清楚指出缺少哪些實作」。一個只寫 `NotImplementedError` 的 stub
    達不到這個條件。
    """


def pending(stage: str, sdd: str, phase: str, todo: list[str]) -> NoReturn:
    body = "\n".join(f"      - {t}" for t in todo)
    raise StageNotImplemented(
        f"\n  階段 {stage} 尚未實作（SDD {sdd}，屬 {phase}）\n"
        f"    待實作：\n{body}\n"
    )
