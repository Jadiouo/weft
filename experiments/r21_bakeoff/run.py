"""S4a 選型對打：同一批代表幀、同一份 prompt、同一套評分。

    conda run -n pipe-cpu python -m experiments.r21_bakeoff.run [模型...]

**條件與 R16 的去重架構一致**——只送第 1、2 道篩選後留下的 21 張，
而不是逐區段的 49 張。這才是實際會跑的條件。

評分素材是 `r14_local_vlm/ground_truth.py` 的 11 張人工繕打投影片。
"""
from __future__ import annotations
import base64, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.r16_slide_dedup.prototype import (  # noqa: E402
    SCHEMA, STUDIO_DISTANCE, SYSTEM, cluster, pick_representative, studio_distances,
)

import requests  # noqa: E402

HERE = Path(__file__).parent
WORK = Path("work/zIglvjoU9vo")
OLLAMA = "http://localhost:11434/api/chat"


#: JSON 解析失敗時重試幾次。**這不是為了讓數字好看**——S4 本來就有
#: `with_retries`，量測條件必須與實際會跑的條件一致。失敗次數另外記錄。
MAX_RETRIES = 2


def ask(model: str, sid: str) -> tuple[dict | None, float, int, str]:
    """回傳 (解析後的 dict 或 None, 總耗時, 重試次數, 最後一次的原始輸出)。"""
    img = base64.b64encode((WORK / "03_slides" / f"{sid}.png").read_bytes()).decode()
    t0 = time.time()
    raw = ""
    for attempt in range(MAX_RETRIES + 1):
        r = requests.post(OLLAMA, timeout=1800, json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "這是不是投影片？若是，逐字轉錄並描述版面。",
                 "images": [img]},
            ],
            "format": SCHEMA, "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
        })
        r.raise_for_status()
        raw = r.json()["message"].get("content") or ""
        try:
            return json.loads(raw), time.time() - t0, attempt, raw
        except json.JSONDecodeError:
            continue
    return None, time.time() - t0, MAX_RETRIES, raw


def main(argv: list[str]) -> int:
    models = argv or ["gemma3:12b", "gemma4:12b", "qwen3-vl:8b", "qwen2.5vl:7b"]
    truth = json.loads(
        (HERE.parent / "r14_image_binding" / "labels_is_slide.json").read_text())
    all_ids = sorted(truth)

    dist = studio_distances(all_ids)
    kept = [s for s in all_ids if dist[s] >= STUDIO_DISTANCE]
    reps = [pick_representative(g) for g in cluster(kept)]
    print(f"送 VLM 的代表幀：{len(all_ids)} → {len(reps)}\n")

    for model in models:
        out_path = HERE / f"out_{model.replace(':', '_')}.json"
        rows = json.loads(out_path.read_text()) if out_path.exists() else {}
        todo = [s for s in reps if s not in rows]
        print(f"=== {model} ===（已有 {len(rows)}／{len(reps)}）", flush=True)
        for sid in todo:
            try:
                parsed, dt, retries, raw = ask(model, sid)
            except Exception as exc:  # noqa: BLE001
                print(f"  {sid} 失敗：{type(exc).__name__} {str(exc)[:70]}", flush=True)
                rows[sid] = {"parsed": None, "sec": 0.0, "retries": MAX_RETRIES,
                             "error": f"{type(exc).__name__}: {exc}"[:200]}
                out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
                continue
            rows[sid] = {"parsed": parsed, "sec": round(dt, 1), "retries": retries}
            if parsed is None:
                rows[sid]["raw"] = raw[:600]
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
            pred = "?" if parsed is None else ("片" if parsed.get("is_slide") else "—")
            exp = "片" if truth[sid] else "—"
            n = 0 if parsed is None else len(parsed.get("slide_text") or "")
            print(f"  {sid} 真值{exp} 預測{pred} {n:4d}字 {dt:6.1f}s"
                  f"{'' if pred == exp else '   ← 不符'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
