"""跨集選型對打：把 R21 的比較擴到播放清單的另外三集。

    conda run -n pipe-cpu python -m experiments.r21_bakeoff.cross

R21 §6 記的第 3 項限制：標準答案 12 張全來自第 1 集，
經脈圖、臟腑解剖、表格這些版面完全沒進來。這裡補上。
"""
from __future__ import annotations
import base64, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.r14_local_vlm.ground_truth import CROSS_EPISODE  # noqa: E402
from experiments.r16_slide_dedup.prototype import SCHEMA, SYSTEM  # noqa: E402
from experiments.r21_bakeoff.score import cer, norm  # noqa: E402

import requests  # noqa: E402

HERE = Path(__file__).parent
OLLAMA = "http://localhost:11434/api/chat"
MAX_RETRIES = 2

#: 每個模型的 context 長度。**不是為了讓誰好看而個別調**——
#: gemma4:12b 在 `num_ctx=8192` 下對長文投影片回傳**空內容**
#: （`done_reason=None`、`eval_count=None`，即請求根本沒完成），
#: 16384 就正常。它的影像 token 比 gemma3 多，8192 不夠放
#: 「圖 + prompt + 長輸出」。這是環境參數不是能力差異，
#: 用不足的 ctx 去比等於在比「誰的圖比較小」。
NUM_CTX = {"gemma4:12b": 16384}
DEFAULT_CTX = 8192


def ask(model: str, path: Path) -> tuple[dict | None, float]:
    img = base64.b64encode(path.read_bytes()).decode()
    t0 = time.time()
    for _ in range(MAX_RETRIES + 1):
        r = requests.post(OLLAMA, timeout=1800, json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "這是不是投影片？若是，逐字轉錄並描述版面。",
                 "images": [img]},
            ],
            "format": SCHEMA, "stream": False,
            "options": {"temperature": 0.2,
                        "num_ctx": NUM_CTX.get(model, DEFAULT_CTX)},
        })
        r.raise_for_status()
        try:
            return json.loads(r.json()["message"].get("content") or ""), time.time() - t0
        except json.JSONDecodeError:
            continue
    return None, time.time() - t0


def main(argv: list[str]) -> int:
    models = argv or ["gemma3:12b", "gemma4:12b", "qwen2.5vl:7b"]
    out_path = HERE / "cross_results.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else {}

    targets = [(vid, sid) for vid, slides in CROSS_EPISODE.items() for sid in slides]
    for model in models:
        for vid, sid in targets:
            key = f"{model}|{vid}|{sid}"
            if key in rows:
                continue
            parsed, dt = ask(model, Path(f"work/{vid}/03_slides/{sid}.png"))
            rows[key] = {"parsed": parsed, "sec": round(dt, 1)}
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
            ref = CROSS_EPISODE[vid][sid]
            v = cer((parsed or {}).get("slide_text") or "", ref) if parsed else 1.0
            print(f"  {model:14s} {vid[:11]}/{sid} CER {v:6.1%}  {dt:5.1f}s", flush=True)

    print("\n" + "=" * 70)
    print("跨集 CER")
    print("=" * 70)
    head = "".join(f"{m.split(':')[0][:9]:>11s}" for m in models)
    print(f"{'集/投影片':26s} {'字數':>5s}{head}")
    tot = {m: [] for m in models}
    for vid, sid in targets:
        ref = CROSS_EPISODE[vid][sid]
        cells = ""
        for m in models:
            p = (rows.get(f"{m}|{vid}|{sid}") or {}).get("parsed")
            v = cer((p or {}).get("slide_text") or "", ref) if p else 1.0
            tot[m].append(v)
            cells += f"{v:10.1%} "
        print(f"{vid[:11]+'/'+sid:26s} {len(norm(ref)):5d}{cells}")
    print("─" * 70)
    print(f"{'平均':26s} {'':5s}" +
          "".join(f"{sum(tot[m]) / len(tot[m]):10.1%} " for m in models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
