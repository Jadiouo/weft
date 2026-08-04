"""A/B 測試：`is_slide` 放在 `description` 之前 vs 之後。

    conda run -n pipe-cpu python -m experiments.r16_slide_dedup.order_ab [模型]

**假設**：structured output 是逐欄生成的，先產生的欄位不能因後面的內容而
回頭修改（SDD §4.7、known-risks R9 的緩解措施就建立在這個性質上）。
現行 schema 把 `is_slide` 排第一，模型在「還沒用文字描述過這張圖」的狀態下
就得先分類。

**證據**：12B 的誤報中，模型自己在 `description` 裡寫了
「背景是講者在佈景板前演講的場景」「畫面為講者與背景文字的組合」——
它看得出來，只是分類時還沒把話說出口。

若假設成立，把 `description` 排到 `is_slide` 之前應該提升分類準確率。
若沒有提升，就是假設錯了，照實寫。
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
WORK = Path("work/zIglvjoU9vo")
OLLAMA = "http://localhost:11434/api/chat"

from .prototype import SYSTEM  # noqa: E402

#: A：現行順序——先分類，再描述
SCHEMA_CLASSIFY_FIRST = {
    "type": "object",
    "properties": {
        "is_slide": {"type": "boolean"},
        "reject_reason": {"type": "string"},
        "slide_text": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["is_slide", "reject_reason", "slide_text", "description"],
}

#: B：先描述看到什麼，再分類
SCHEMA_DESCRIBE_FIRST = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "is_slide": {"type": "boolean"},
        "reject_reason": {"type": "string"},
        "slide_text": {"type": "string"},
    },
    "required": ["description", "is_slide", "reject_reason", "slide_text"],
}

SYSTEM_B = SYSTEM.replace(
    "### 1. 判斷這張圖是不是投影片（`is_slide`）",
    "### 先描述你看到什麼（`description`），再判斷這張圖是不是投影片（`is_slide`）",
)


def ask(model: str, sid: str, schema: dict, system: str) -> tuple[dict | None, float]:
    img = base64.b64encode((WORK / "03_slides" / f"{sid}.png").read_bytes()).decode()
    t0 = time.time()
    r = requests.post(OLLAMA, timeout=1800, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "這是不是投影片？若是，逐字轉錄並描述版面。",
             "images": [img]},
        ],
        "format": schema, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    })
    dt = time.time() - t0
    r.raise_for_status()
    try:
        return json.loads(r.json()["message"]["content"]), dt
    except json.JSONDecodeError:
        return None, dt


def main(argv: list[str]) -> int:
    model = argv[0] if argv else "gemma3:12b"
    truth = json.loads(
        (HERE.parent / "r14_image_binding" / "labels_is_slide.json").read_text())
    # 只測第 1、2 道之後留下的那 21 張——那才是實際會送 VLM 的集合
    reps = sorted(json.loads((HERE / "out_gemma3_12b.json").read_text()))

    out_path = HERE / f"ab_{model.replace(':', '_')}.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else {}

    for arm, schema, system in (("A_classify_first", SCHEMA_CLASSIFY_FIRST, SYSTEM),
                                ("B_describe_first", SCHEMA_DESCRIBE_FIRST, SYSTEM_B)):
        print(f"\n=== {arm} ===", flush=True)
        for sid in reps:
            key = f"{arm}:{sid}"
            if key in rows:
                continue
            parsed, dt = ask(model, sid, schema, system)
            rows[key] = {"parsed": parsed, "sec": round(dt, 1)}
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
            pred = "?" if parsed is None else ("片" if parsed.get("is_slide") else "—")
            exp = "片" if truth[sid] else "—"
            print(f"  {sid} 真值{exp} 預測{pred} {dt:5.1f}s"
                  f"{'' if pred == exp else '   ← 不符'}", flush=True)

    print("\n" + "=" * 56)
    for arm in ("A_classify_first", "B_describe_first"):
        ok = fp = fn = 0
        for sid in reps:
            p = (rows.get(f"{arm}:{sid}") or {}).get("parsed")
            if p is None:
                continue
            pred, exp = bool(p.get("is_slide")), truth[sid]
            ok += pred == exp
            fp += pred and not exp
            fn += (not pred) and exp
        print(f"{arm:20s} 準確率 {ok / len(reps):5.1%}  誤報 {fp}  漏抓 {fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
