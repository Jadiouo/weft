"""temperature 0 下的轉錄對打——**這才是生產設定**。

    conda run -n pipe-cpu python -m experiments.r21_bakeoff.temp0

R21 第一輪用 `temperature=0.2` 且**每張只跑一次**。實測發現那對高變異的
模型不可靠：qwen2.5vl 在 `slide_029`（內觀經）上第一輪量到 10.1%，
重跑是 **36.7%**（temp 0）或 **57.4%**（temp 0.2）——第一輪抽到了好籤。

R14 早就量過 qwen2.5vl 的自我一致性只有 18.0%，第一輪對單次量測給了
太多權重。這一輪固定 temperature 0（S4a 的實際設定），涵蓋全部 16 張。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiments.r14_local_vlm.ground_truth import all_ground_truth  # noqa: E402
from experiments.r21_bakeoff.score import cer, norm  # noqa: E402
from weft.stages.providers import Part, generate  # noqa: E402
from weft.stages.slides import RESPONSE_SCHEMA, SYSTEM_PROMPT, USER_PROMPT  # noqa: E402

HERE = Path(__file__).parent
MODELS = ["ollama:qwen2.5vl:7b", "ollama:gemma3:12b", "ollama:gemma4:12b"]


def main() -> int:
    truth = all_ground_truth()
    out_path = HERE / "temp0_results.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else {}

    for spec in MODELS:
        for (vid, sid), ref in sorted(truth.items()):
            key = f"{spec}|{vid}|{sid}"
            if key in rows:
                continue
            img = Path(f"work/{vid}/03_slides/{sid}.png")
            if not img.exists():
                continue
            t0 = time.time()
            try:
                r = generate(spec, SYSTEM_PROMPT,
                             [Part(text=USER_PROMPT), Part(image=img.read_bytes())],
                             RESPONSE_SCHEMA, temperature=0.0)
                text = r.payload.get("slide_text") or ""
            except Exception as exc:  # noqa: BLE001
                text = ""
                print(f"  {spec} {sid} 失敗：{str(exc)[:60]}", flush=True)
            rows[key] = {"slide_text": text, "sec": round(time.time() - t0, 1)}
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
            print(f"  {spec:22s} {vid[:11]}/{sid} CER {cer(text, ref):6.1%}", flush=True)

    print("\n" + "=" * 78)
    print("temperature 0 下的 slide_text CER")
    print("=" * 78)
    head = "".join(f"{m.split(':')[1][:9]:>12s}" for m in MODELS)
    print(f"{'集/投影片':26s} {'字數':>5s}{head}")
    tot = {m: [] for m in MODELS}
    for (vid, sid), ref in sorted(truth.items()):
        cells = ""
        have = False
        for m in MODELS:
            r = rows.get(f"{m}|{vid}|{sid}")
            if r is None:
                cells += f"{'—':>12s}"
                continue
            have = True
            v = cer(r["slide_text"], ref)
            tot[m].append(v)
            cells += f"{v:11.1%} "
        if have:
            print(f"{vid[:11] + '/' + sid:26s} {len(norm(ref)):5d}{cells}")
    print("─" * 78)
    for label, fn in (("平均", lambda v: sum(v) / len(v)),
                      ("中位", lambda v: sorted(v)[len(v) // 2])):
        print(f"{label:26s} {'':5s}" +
              "".join(f"{fn(tot[m]):11.1%} " if tot[m] else f"{'—':>12s}" for m in MODELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
