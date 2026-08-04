"""量測本地 VLM 能否取代 Gemini Flash Lite 做 S4 的前兩項工作。

    conda run -n pipe-cpu python -m experiments.r14_local_vlm.run [模型...]

只量 S4 的**第 1、2 項工作**（`is_slide` 與 `slide_text`）——那兩項是
整條溯源鏈的地基。第 3、4 項（校正、理解）建立在它們之上，地基不穩就不必談。

**一張圖一次呼叫。** 本地沒有額度壓力，批次的唯一動機（省額度）不存在；
順帶完全避開 D20 的圖片↔區段錯位問題。這本身就是本地方案的一個優點。

標準答案：`../r14_image_binding/labels_is_slide.json`（人工逐張判讀）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
LABELS = HERE.parent / "r14_image_binding" / "labels_is_slide.json"
SLIDES = Path("work/zIglvjoU9vo/03_slides")
OLLAMA = "http://localhost:11434/api/chat"

#: 取自 src/weft/stages/understand.py SYSTEM_PROMPT 的第 1、2 項工作，
#: 一字未改——改了就不是在量「能不能取代」而是在量「另一個任務」。
SYSTEM = """你在為一個「講經影片 → 可檢索知識庫」的系統做內容理解。
你會拿到一段影片的**代表畫面**。

代表畫面是自動抽出的——系統只知道「這段時間畫面是靜止的」，**不知道那是
投影片還是講者鏡頭**。判斷它是什麼，是你的第一項工作。

### 1. 判斷這張圖是不是投影片（`is_slide`）

**是投影片**：畫面主體為文字、圖表、經文、流程圖等準備好的教材內容。
**不是投影片**：講者的攝影棚鏡頭、片頭片尾動畫、純裝飾畫面。

注意：講者所在的攝影棚背景**經常有大量裝飾文字**（標語、書法、招牌）。
那些是**佈景**，不是投影片內容。判斷依據是「這是為了講解而製作的教材」，
不是「畫面上有沒有字」。

`is_slide: false` 時，填 `reject_reason`（一句話），`slide_text` 留空字串。

### 2. 逐字轉錄投影片上的文字（`slide_text`）

**先抄，再詮釋。** 把畫面上的文字**原樣**打出來，保留換行與排列順序
（直排請由右至左、由上而下）。這一欄是後續溯源檢查的比對基準，
**不要在這裡改寫、摘要或補充**。

`is_slide: false` 時填空字串。

只輸出 JSON，不要加任何說明文字。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "is_slide": {"type": "boolean"},
        "reject_reason": {"type": "string"},
        "slide_text": {"type": "string"},
    },
    "required": ["is_slide", "reject_reason", "slide_text"],
}


def ask(model: str, image_b64: str, timeout: int = 600) -> tuple[dict | None, float, str]:
    """回傳 (解析後的 dict 或 None, 耗時秒數, 原始回應)。"""
    t0 = time.time()
    r = requests.post(OLLAMA, timeout=timeout, json={
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "這張圖是不是投影片？若是，逐字轉錄上面的文字。",
             "images": [image_b64]},
        ],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    })
    dt = time.time() - t0
    r.raise_for_status()
    raw = r.json()["message"]["content"]
    try:
        return json.loads(raw), dt, raw
    except json.JSONDecodeError:
        return None, dt, raw


def main(argv: list[str]) -> int:
    import base64

    models = argv or ["gemma3:4b", "gemma3:12b", "qwen2.5vl:7b"]
    truth = json.loads(LABELS.read_text())
    slide_ids = sorted(truth)

    images = {sid: base64.b64encode((SLIDES / f"{sid}.png").read_bytes()).decode()
              for sid in slide_ids}

    for model in models:
        out_path = HERE / f"raw_{model.replace(':', '_').replace('/', '_')}.json"
        rows = json.loads(out_path.read_text()) if out_path.exists() else {}
        print(f"\n=== {model} ===（已有 {len(rows)}／{len(slide_ids)}）", flush=True)

        for i, sid in enumerate(slide_ids):
            if sid in rows:
                continue
            try:
                parsed, dt, raw = ask(model, images[sid])
            except Exception as exc:  # noqa: BLE001
                print(f"  {sid} 失敗：{type(exc).__name__} {str(exc)[:80]}", flush=True)
                rows[sid] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
                continue

            rows[sid] = {
                "parsed": parsed,
                "raw_len": len(raw),
                "json_ok": parsed is not None,
                "sec": round(dt, 2),
            }
            ok = "?" if parsed is None else ("片" if parsed.get("is_slide") else "—")
            exp = "片" if truth[sid] else "—"
            n = 0 if parsed is None else len(parsed.get("slide_text") or "")
            print(f"  [{i + 1:2d}/{len(slide_ids)}] {sid} 真值{exp} 預測{ok} "
                  f"{n:4d}字 {dt:5.1f}s{'' if ok == exp else '   ← 不符'}", flush=True)
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
