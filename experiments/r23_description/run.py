"""R23：`description` 的把關訊號——多模型一致性有沒有鑑別力？

    conda run -n pipe-cpu python -m experiments.r23_description.run

**背景**：R16 §4 記的洞——`description` 會編造，而 R12 量到 `圖表描述`
型別的溯源分離度是 **0.00x**，現有機制一個都抓不到。

**要測的假設**（沿用 R19 的作法）：讓多個模型各自描述同一張投影片，
**在編造處它們會各說各話，在真實處會一致**。

判準比照 D1／R12：分離倍數 <2x 即判該訊號不可用，不調參數硬拉。
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from weft.stages.providers import Part, generate  # noqa: E402
from weft.stages.slides import RESPONSE_SCHEMA, SYSTEM_PROMPT, USER_PROMPT  # noqa: E402

HERE = Path(__file__).parent
MODELS = ["ollama:gemma3:12b", "ollama:gemma4:12b", "ollama:qwen2.5vl:7b"]

#: 我逐張看過原圖後列出的**事實**與**常見編造**。
#: 「事實」是畫面上確實有的；「編造」是模型說過但畫面上沒有的。
CLAIMS: dict[str, dict[str, list[str]]] = {
    "zIglvjoU9vo/slide_029": {
        "fact": ["綠色", "箭頭", "時間軸", "色身漸完成", "功能漸齊備"],
        "fabricated": ["黃色", "表格", "藍色", "長條圖", "照片"],
    },
    "zIglvjoU9vo/slide_045": {
        # **標註修正**：原本把「藍色」列為編造，被 2/3 個模型提到後回頭
        # 看原圖——右側解剖圖裡的胚胎確實是**藍紫色的環狀構造**。
        # 是我看漏了，不是模型編的。修正後它進「事實」欄。
        # 記在這裡是因為：**對照組出錯時，它會偽裝成「訊號沒有鑑別力」**。
        "fact": ["黃", "解剖", "三魂", "藍色"],
        "fabricated": ["表格", "箭頭", "綠色"],
    },
    "zIglvjoU9vo/slide_041": {
        "fact": ["老鼠", "循環", "胚胎"],
        "fabricated": ["表格", "長條圖", "中文標註"],
    },
    "C_CFyilE-ks/slide_012": {
        "fact": ["黃", "藍", "兩", "陰陽"],
        "fabricated": ["箭頭", "圖片", "表格", "綠色"],
    },
}


def main() -> int:
    out_path = HERE / "descriptions.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else {}

    for spec in MODELS:
        for key in CLAIMS:
            vid, sid = key.split("/")
            cache = f"{spec}|{key}"
            if cache in rows:
                continue
            img = Path(f"work/{vid}/03_slides/{sid}.png")
            try:
                r = generate(spec, SYSTEM_PROMPT,
                             [Part(text=USER_PROMPT), Part(image=img.read_bytes())],
                             RESPONSE_SCHEMA, temperature=0.0)
                rows[cache] = r.payload.get("description") or ""
            except Exception as exc:  # noqa: BLE001
                rows[cache] = ""
                print(f"  {spec} {key} 失敗：{str(exc)[:60]}", flush=True)
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
            print(f"  {spec:22s} {key} {len(rows[cache])} 字", flush=True)

    # ---- 一致性 vs 正確性 ------------------------------------------
    print("\n" + "=" * 72)
    print("每個主張被幾個模型提到（共 3 個）")
    print("=" * 72)
    fact_hits: list[int] = []
    fab_hits: list[int] = []
    for key, groups in CLAIMS.items():
        descs = [rows.get(f"{m}|{key}", "") for m in MODELS]
        print(f"\n{key}")
        for kind, claims in groups.items():
            for claim in claims:
                n = sum(1 for d in descs if claim in d)
                (fact_hits if kind == "fact" else fab_hits).append(n)
                mark = "事實" if kind == "fact" else "編造"
                print(f"  [{mark}] {claim:12s} {n}/3 {'█' * n}")

    print("\n" + "=" * 72)
    if fact_hits and fab_hits:
        f_avg = sum(fact_hits) / len(fact_hits)
        b_avg = sum(fab_hits) / len(fab_hits)
        print(f"事實主張平均被 {f_avg:.2f}/3 個模型提到（n={len(fact_hits)}）")
        print(f"編造主張平均被 {b_avg:.2f}/3 個模型提到（n={len(fab_hits)}）")
        sep = f_avg / b_avg if b_avg > 0 else float("inf")
        print(f"分離倍數 {sep:.2f}x —— "
              f"{'可用' if sep >= 2.0 else '**不可用**（判準同 D1／R12：<2x 即不可用）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
