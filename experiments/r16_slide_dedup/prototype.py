"""S1c 原型：把候選幀縮成「相異投影片」，再逐頁送 VLM。

    conda run -n pipe-cpu python -m experiments.r16_slide_dedup.prototype [模型]

對應 `docs/proposals/v0.4-transcript-first.md` 的第 1、2 步。

三道，由便宜到貴：

1. **砍掉攝影棚定鏡**（純 CV，零成本）。與全片中位幀的距離 <0.10 者剔除。
   實測：攝影棚定鏡 ~0.03、投影片最低 0.281，**分離 6.7x**，零漏抓。
2. **分群去重**（純 CV，零成本）。ink Jaccard 單連結。
   實測門檻 0.20–0.40 結果完全相同，且沒有任何一群混到不同類別。
3. **VLM 逐群一次**。只有這一步花錢／花時間。

第 1 道解決不了片頭片尾與「講者＋大字書法」的標題鏡頭——那些距離也很遠
（0.257–0.349）。**那是刻意留給 VLM 的**：CV 只做它有把握的那一刀。
"""

from __future__ import annotations

import base64
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from weft.stages.detect import ink_jaccard  # noqa: E402
from weft.stages.frames import _ink_mask  # noqa: E402

WORK = Path("work/zIglvjoU9vo")
HERE = Path(__file__).parent
OLLAMA = "http://localhost:11434/api/chat"

#: 與全片中位幀的距離低於此值即視為「攝影棚定鏡」。
#: 實測分離 6.7x（投影片最低 0.281 / 攝影棚最高 0.042），取中間偏保守側。
STUDIO_DISTANCE = 0.10

#: ink Jaccard 單連結的合併門檻。0.20–0.40 實測結果完全相同，取中值。
DEDUP_JACCARD = 0.30

SYSTEM = """你在為一個「講經影片 → 可檢索知識庫」的系統讀投影片。
你會拿到一張從影片中抽出的靜止畫面。

### 1. 判斷這張圖是不是投影片（`is_slide`）

**是投影片**：畫面主體為文字、圖表、經文、流程圖等準備好的教材內容。
**不是投影片**：講者的攝影棚鏡頭、片頭片尾動畫、純裝飾畫面、會場全景。

注意：講者所在的攝影棚背景**經常有大量裝飾文字**（標語、書法、招牌）。
那些是**佈景**，不是投影片內容。判斷依據是「這是為了講解而製作的教材」，
不是「畫面上有沒有字」。

`is_slide: false` 時填 `reject_reason`，其餘欄位留空字串。

### 2. 逐字轉錄（`slide_text`）

**先抄，再詮釋。** 把畫面上的文字**原樣**打出來，保留換行與排列順序。
直排請由右至左、由上而下。多欄並排時，**同一列的左右欄要寫在同一行**——
例如「一月為胞，　精血凝也。」是一整句，不可拆成兩欄分別抄完。

這一欄是後續溯源檢查的比對基準，**不要在這裡改寫、摘要或補充**。

### 3. 版面描述（`description`）

用一段文字說明這一頁的**版面結構與它表達的關係**：箭頭指向什麼、
雙欄如何對應、色彩編碼代表什麼、圖片畫的是什麼。

RAG 讀不到圖，所以「看得懂這張圖的人才知道的事」必須寫成文字。
這一欄可以詮釋，但不得陳述畫面上沒有的資訊。

只輸出 JSON。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "is_slide": {"type": "boolean"},
        "reject_reason": {"type": "string"},
        "slide_text": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["is_slide", "reject_reason", "slide_text", "description"],
}


def studio_distances(slide_ids: list[str]) -> dict[str, float]:
    """每張代表畫面與全片中位幀的平均絕對差（正規化到 0–1）。"""
    frames = sorted(glob.glob(str(WORK / "02_frames" / "*")))
    stack = [cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (160, 90),
                        interpolation=cv2.INTER_AREA) for p in frames[::4]]
    median = np.median(np.stack(stack), axis=0)

    out = {}
    for sid in slide_ids:
        g = cv2.resize(cv2.imread(str(WORK / "03_slides" / f"{sid}.png"),
                                  cv2.IMREAD_GRAYSCALE), (160, 90),
                       interpolation=cv2.INTER_AREA)
        out[sid] = float(np.abs(g.astype(int) - median).mean()) / 255
    return out


def cluster(slide_ids: list[str]) -> list[list[str]]:
    """ink Jaccard 單連結分群。回傳每群的成員（依原順序）。"""
    masks = {}
    for sid in slide_ids:
        g = cv2.imread(str(WORK / "03_slides" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        masks[sid] = _ink_mask(cv2.resize(g, (320, 180), interpolation=cv2.INTER_AREA))

    parent = {s: s for s in slide_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(slide_ids):
        for b in slide_ids[i + 1:]:
            if ink_jaccard(masks[a], masks[b]) < DEDUP_JACCARD:
                parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    for s in slide_ids:
        groups.setdefault(find(s), []).append(s)
    return sorted(groups.values(), key=lambda g: g[0])


def pick_representative(group: list[str]) -> str:
    """取群內 ink 量最大的那張——同 D15，最完整的一幀。"""
    best, best_ink = group[0], -1
    for sid in group:
        g = cv2.imread(str(WORK / "03_slides" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        ink = int(_ink_mask(cv2.resize(g, (320, 180),
                                       interpolation=cv2.INTER_AREA)).sum())
        if ink > best_ink:
            best, best_ink = sid, ink
    return best


def ask(model: str, sid: str, timeout: int = 1800) -> tuple[dict | None, float]:
    img = base64.b64encode((WORK / "03_slides" / f"{sid}.png").read_bytes()).decode()
    t0 = time.time()
    r = requests.post(OLLAMA, timeout=timeout, json={
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": "這是不是投影片？若是，逐字轉錄並描述版面。",
             "images": [img]},
        ],
        "format": SCHEMA, "stream": False,
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
    all_ids = sorted(truth)

    # ---- 第 1 道：砍掉攝影棚定鏡 -------------------------------------
    dist = studio_distances(all_ids)
    kept = [s for s in all_ids if dist[s] >= STUDIO_DISTANCE]
    dropped = [s for s in all_ids if dist[s] < STUDIO_DISTANCE]
    lost = [s for s in dropped if truth[s]]
    print(f"第 1 道（攝影棚定鏡，距離 <{STUDIO_DISTANCE}）："
          f"剔除 {len(dropped)}／{len(all_ids)}，其中誤刪投影片 **{len(lost)}** 張")
    print(f"  剔除者距離 max={max(dist[s] for s in dropped):.3f}；"
          f"留下的投影片距離 min={min(dist[s] for s in kept if truth[s]):.3f}")

    # ---- 第 2 道：分群去重 -------------------------------------------
    groups = cluster(kept)
    reps = [(pick_representative(g), g) for g in groups]
    print(f"\n第 2 道（分群去重）：{len(kept)} 張 → **{len(groups)} 群**")
    for rep, g in reps:
        mark = "片" if truth[rep] else "—"
        extra = f"（代表 {len(g)} 張）" if len(g) > 1 else ""
        print(f"  {rep} 真值{mark} {extra}")

    print(f"\n→ 需要 VLM 的畫面：{len(all_ids)} → **{len(reps)}**"
          f"（真投影片 {sum(1 for r, _ in reps if truth[r])} 群）")

    # ---- 第 3 道：VLM 逐群一次 ---------------------------------------
    out_path = HERE / f"out_{model.replace(':', '_')}.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else {}
    print(f"\n第 3 道（{model}）：")
    for rep, g in reps:
        if rep in rows:
            continue
        parsed, dt = ask(model, rep)
        rows[rep] = {"parsed": parsed, "sec": round(dt, 1), "members": g}
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        pred = "?" if parsed is None else ("片" if parsed.get("is_slide") else "—")
        exp = "片" if truth[rep] else "—"
        n = 0 if parsed is None else len(parsed.get("slide_text") or "")
        print(f"  {rep} 真值{exp} 預測{pred} 轉錄{n:4d}字 {dt:6.1f}s"
              f"{'' if pred == exp else '   ← 不符'}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
